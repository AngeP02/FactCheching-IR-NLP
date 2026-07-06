"""
pipeline.py
════════════════════════════════════════════════════════════════════
Orchestratore della pipeline fact-checking end-to-end.

Responsabilità:
    Coordinare tutti i moduli del sistema in una sequenza coerente
    per verificare una claim testuale.

Flusso della pipeline:
    1. Preprocessing query
       └── Preprocessore.elabora(claim)

    2. Espansione query (opzionale)
       └── EspansoreQuery.espandi(claim)

    3. Recupero documenti
       └── CalcolatoreRanking.recupera_top_k(claim)
       └── Sorgente: CorpusFever (FEVER) o CorpusWeb (web)

    4. Classificazione NLI
       └── ClassificatoreNLI.verifica(claim, documenti)

    5. Restituzione risultato strutturato

Modalità operative:
    La pipeline supporta due sorgenti di corpus intercambiabili
    grazie all'interfaccia comune CorpusBase:
    - 'fever': usa il corpus FEVER pre-costruito e l'indice su disco
    - 'web':   recupera documenti Wikipedia in tempo reale

    Il cambio di modalità avviene modificando SORGENTE_CORPUS
    in configurazione.py — nessuna modifica al codice della pipeline.

Separazione valutazione vs inferenza:
    - verifica():  modalità inferenza — una singola claim
    - valuta():    modalità valutazione — un batch di claim con
                   ground truth, produce metriche IR e NLP

Utilizzo:
    from sorgente.pipeline import Pipeline

    pipeline = Pipeline(api_key='...')

    # Inferenza
    risultato = pipeline.verifica('Obama was born in Hawaii')

    # Valutazione
    metriche = pipeline.valuta(df_claim_con_ground_truth)
════════════════════════════════════════════════════════════════════
"""

import os
import pickle

import config as cfg
from Sorgente.Preprocessing.preprocesing        import Preprocessore
from Sorgente.Espansione.espansione_query          import EspansoreQuery
from Sorgente.Indicizzazione.indice_invertito     import IndiceInvertito
from Sorgente.Recupero.metadati                   import GestoreMetadati
from Sorgente.Recupero.bm25                       import CalcolatoreRanking
from Sorgente.Classificazione.classificatore_nli  import ClassificatoreNLI
from Sorgente.Valutazione.metriche_ir             import MetricheIR
from Sorgente.Valutazione.metriche_nlp            import MetricheNLP


class Pipeline:
    """
    Orchestratore della pipeline fact-checking end-to-end.

    Inizializza e coordina tutti i moduli del sistema.
    Espone due metodi principali:
        verifica() → per l'inferenza su una singola claim
        valuta()   → per la valutazione su un batch con ground truth

    Attributes:
        _preprocessore:    preprocessore testuale
        _espansore:        espansore query WordNet
        _indice:           indice invertito del corpus
        _corpus:           dizionario id_doc → dati documento
        _metadati:         gestore authority e freshness score
        _ranking:          calcolatore BM25 esteso
        _classificatore:   classificatore NLI via LLM
        _metriche_ir:      calcolatore metriche IR
        _metriche_nlp:     calcolatore metriche NLP
    """

    def __init__(self, api_key: str):
        """
        Inizializza la pipeline caricando tutti i moduli.

        Il caricamento avviene una volta sola — tutti i moduli
        vengono tenuti in memoria per tutta la sessione.
        I percorsi e i parametri vengono letti da configurazione.py.

        Args:
            api_key: chiave API OpenAI per il classificatore NLI
        """
        print('Inizializzazione pipeline...')

        # Preprocessing e espansione query
        self._preprocessore = Preprocessore()
        self._espansore     = EspansoreQuery(self._preprocessore)
        print('  ✓ Preprocessore e espansore query')

        # Corpus e indice — dipende dalla modalità (fever o web)
        self._corpus = self._carica_corpus()
        self._indice = self._carica_indice()
        print(f'  ✓ Corpus ({len(self._corpus):,} documenti) e indice '
              f'({self._indice.numero_termini:,} termini)')

        # Metadati e ranking
        self._metadati = GestoreMetadati(self._corpus)
        self._ranking  = CalcolatoreRanking(
            self._indice,
            self._preprocessore,
            self._metadati
        )
        print('  ✓ Metadati e ranking BM25 esteso')

        # Classificatore NLI
        self._classificatore = ClassificatoreNLI(api_key)
        print('  ✓ Classificatore NLI')

        # Metriche
        self._metriche_ir  = MetricheIR()
        self._metriche_nlp = MetricheNLP()
        print('  ✓ Moduli metriche')

        print('Pipeline pronta.\n')

    # ── Caricamento risorse ───────────────────────────────────────

    def _carica_corpus(self) -> dict:
        """
        Carica il corpus in base alla modalità configurata.

        In modalità 'fever': carica corpus_fever_arricchito.pkl
        In modalità 'web':   restituisce dizionario vuoto
                             (il corpus web viene costruito a runtime)

        Returns:
            Dizionario {id_doc: {lines, text, numero_link_entrata, timestamp}}
        """
        if cfg.SORGENTE_CORPUS == 'fever':
            if not os.path.exists(cfg.PERCORSO_CORPUS_ARRICCHITO):
                raise FileNotFoundError(
                    f"Corpus non trovato: {cfg.PERCORSO_CORPUS_ARRICCHITO}\n"
                    "Esegui prima il notebook 00_studio_dataset.ipynb"
                )
            with open(cfg.PERCORSO_CORPUS_ARRICCHITO, 'rb') as f:
                return pickle.load(f)
        else:
            # Modalità web: corpus vuoto, i documenti arrivano a runtime
            return {}

    def _carica_indice(self) -> IndiceInvertito:
        """
        Carica l'indice invertito da disco.

        In modalità 'fever': carica indice_fever.pkl
        In modalità 'web':   restituisce un indice vuoto
                             (l'indice web viene costruito a runtime)

        Returns:
            Istanza di IndiceInvertito pronta per le query
        """
        indice = IndiceInvertito()

        if cfg.SORGENTE_CORPUS == 'fever':
            if not os.path.exists(cfg.PERCORSO_INDICE):
                raise FileNotFoundError(
                    f"Indice non trovato: {cfg.PERCORSO_INDICE}\n"
                    "Esegui prima il notebook 02_test_indice.ipynb"
                )
            indice.carica(cfg.PERCORSO_INDICE)

        return indice

    # ── Interfaccia pubblica ──────────────────────────────────────

    def verifica(
        self,
        claim:           str,
        sentence_ids_map: dict | None = None
    ) -> dict:
        """
        Verifica una singola claim attraverso la pipeline completa.

        Passi eseguiti:
            1. Espansione query (se abilitata in configurazione.py)
            2. Recupero top-K documenti con BM25 esteso
            3. Classificazione NLI per ogni documento
            4. Aggregazione verdetti con maggioranza pesata

        Args:
            claim:            testo della claim da verificare
            sentence_ids_map: {id_doc: [sentence_ids]} per frasi specifiche
                              Usato solo in modalità 'fever' con ground truth

        Returns:
            Dizionario con:
                'claim':        testo originale
                'verdetto':     'SUPPORTS' / 'REFUTES' / 'NOT ENOUGH INFO'
                'confidenza':   percentuale di confidenza del verdetto
                'pesi_label':   pesi aggregati per ogni label
                'documenti':    lista documenti usati con i rispettivi verdetti
                'token_query':  token dopo preprocessing (+ espansione se attiva)
        """
        # Passo 1 — Preprocessing + espansione query
        # L'espansore è trasparente: se disabilitato restituisce solo
        # i token originali senza modifiche
        token_query = self._espansore.espandi(claim)

        # Passo 2 — Recupero documenti
        # In modalità web il ranking usa documenti recuperati in tempo reale
        if cfg.SORGENTE_CORPUS == 'web':
            risultati_bm25, corpus_runtime = self._recupera_da_web(claim)
            corpus_da_usare = corpus_runtime
        else:
            risultati_bm25  = self._ranking.recupera_top_k(
                claim, k=cfg.DOCUMENTI_DA_RECUPERARE
            )
            corpus_da_usare = self._corpus

        if not risultati_bm25:
            return {
                'claim':       claim,
                'verdetto':    'NOT ENOUGH INFO',
                'confidenza':  0.0,
                'pesi_label':  {},
                'documenti':   [],
                'token_query': token_query
            }

        # Passo 3+4 — Classificazione NLI e aggregazione
        risultato_nli = self._classificatore.verifica(
            claim             = claim,
            risultati_bm25    = risultati_bm25,
            corpus_arricchito = corpus_da_usare,
            sentence_ids_map  = sentence_ids_map
        )

        return {
            'claim':       claim,
            'verdetto':    risultato_nli['verdetto'],
            'confidenza':  risultato_nli['confidenza'],
            'pesi_label':  risultato_nli['pesi_label'],
            'documenti':   risultato_nli['dettagli'],
            'token_query': token_query
        }

    def valuta(
        self,
        claim_con_ground_truth: list[dict],
        k_ir:                   int = cfg.DOCUMENTI_DA_RECUPERARE
    ) -> dict:
        """
        Valuta il sistema su un batch di claim con ground truth.

        Calcola sia le metriche IR (qualità del recupero) che le
        metriche NLP (qualità della classificazione).

        Nota: le claim NOT ENOUGH INFO vengono incluse nella
        valutazione NLP ma escluse dalla valutazione IR perché
        non hanno un documento rilevante da recuperare.

        Args:
            claim_con_ground_truth: lista di dizionari:
                {
                    'claim':         str,
                    'label':         str,          # ground truth
                    'url_documenti': list[str],    # documenti rilevanti
                    'sentence_ids':  list[int]     # frasi rilevanti
                }
            k_ir: numero di documenti da recuperare per la valutazione IR

        Returns:
            Dizionario con:
                'metriche_ir':  risultati MetricheIR.valuta_sistema()
                'metriche_nlp': risultati MetricheNLP.valuta_sistema()
                'n_claim':      numero totale di claim valutate
                'n_claim_ir':   numero di claim usate per IR
                                (esclude NOT ENOUGH INFO)
        """
        risultati_ir  = []
        label_vere    = []
        label_predette = []

        print(f'Valutazione su {len(claim_con_ground_truth)} claim...')

        for i, item in enumerate(claim_con_ground_truth):
            claim        = item['claim']
            label_vera   = item['label']
            doc_rilevanti = set(item.get('url_documenti', []))
            sids         = item.get('sentence_ids', [])

            # Mappa sentence_ids per il classificatore
            sentence_ids_map = {}
            for url, sid in zip(item.get('url_documenti', []), sids):
                if url not in sentence_ids_map:
                    sentence_ids_map[url] = []
                sentence_ids_map[url].append(sid)

            # Recupero documenti
            risultati_bm25 = self._ranking.recupera_top_k(claim, k=k_ir)
            id_recuperati  = [r['id_doc'] for r in risultati_bm25]

            # Valutazione IR — solo per claim con documenti rilevanti
            if doc_rilevanti:
                risultati_ir.append({
                    'recuperati': id_recuperati,
                    'rilevanti':  doc_rilevanti
                })

            # Classificazione NLI
            if risultati_bm25:
                risultato_nli = self._classificatore.verifica(
                    claim             = claim,
                    risultati_bm25    = risultati_bm25,
                    corpus_arricchito = self._corpus,
                    sentence_ids_map  = sentence_ids_map if sentence_ids_map else None
                )
                label_predetta = risultato_nli['verdetto']
            else:
                label_predetta = 'NOT ENOUGH INFO'

            label_vere.append(label_vera)
            label_predette.append(label_predetta)

            # Progresso ogni 10 claim
            if (i + 1) % 10 == 0:
                print(f'  {i+1}/{len(claim_con_ground_truth)} claim elaborate...')

        # Calcolo metriche finali
        metriche_ir  = self._metriche_ir.valuta_sistema(risultati_ir)
        metriche_nlp = self._metriche_nlp.valuta_sistema(label_vere, label_predette)

        return {
            'metriche_ir':  metriche_ir,
            'metriche_nlp': metriche_nlp,
            'n_claim':      len(claim_con_ground_truth),
            'n_claim_ir':   len(risultati_ir)
        }

    # ── Modalità web ──────────────────────────────────────────────

    def _recupera_da_web(self, claim: str) -> tuple[list[dict], dict]:
        """
        Recupera documenti Wikipedia in tempo reale per una claim.

        Usato in modalità SORGENTE_CORPUS = 'web'.
        Costruisce un mini-corpus e un mini-indice al volo,
        poi applica BM25 sui documenti recuperati.

        Args:
            claim: testo della claim

        Returns:
            Tupla (risultati_bm25, corpus_runtime):
                risultati_bm25:  lista documenti con punteggi
                corpus_runtime:  dizionario id_doc → dati documento
        """
        # Import qui per non caricare wikipediaapi in modalità fever
        from Sorgente.Corpus.Corpus_web import CorpusWeb

        corpus_web     = CorpusWeb(self._preprocessore)
        corpus_runtime = corpus_web.ottieni_documenti(claim)

        if not corpus_runtime:
            return [], {}

        # Costruisce indice temporaneo sui documenti recuperati
        from Sorgente.Indicizzazione.indice_invertito import IndiceInvertito
        indice_temp = IndiceInvertito()
        for id_doc, doc in corpus_runtime.items():
            token = self._preprocessore.elabora(doc.get('text', ''))
            indice_temp.aggiungi_documento(id_doc, token)

        # Metadati temporanei
        metadati_temp = GestoreMetadati(corpus_runtime)

        # Ranking sul corpus temporaneo
        ranking_temp = CalcolatoreRanking(
            indice_temp,
            self._preprocessore,
            metadati_temp
        )
        risultati = ranking_temp.recupera_top_k(
            claim, k=cfg.DOCUMENTI_DA_RECUPERARE
        )
        return risultati, corpus_runtime

    # ── Utilità ───────────────────────────────────────────────────

    def info(self) -> None:
        """
        Stampa un riepilogo della configurazione attuale della pipeline.
        """
        print('CONFIGURAZIONE PIPELINE')
        print('=' * 45)
        print(f'  Sorgente corpus:    {cfg.SORGENTE_CORPUS}')
        print(f'  Documenti top-K:    {cfg.DOCUMENTI_DA_RECUPERARE}')
        print(f'  BM25 k1:            {cfg.K1}')
        print(f'  BM25 b:             {cfg.B}')
        print(f'  Peso authority:     {cfg.PESO_AUTHORITY}')
        print(f'  Peso freshness:     {cfg.PESO_FRESHNESS}')
        print(f'  Espansione attiva:  {cfg.ESPANSIONE_ATTIVA}')
        print(f'  Modello LLM:        {cfg.MODELLO_LLM}')
        print(f'  Temperatura LLM:    {cfg.TEMPERATURA}')
        print(f'  Documenti corpus:   {len(self._corpus):,}')
        print(f'  Termini indice:     {self._indice.numero_termini:,}')