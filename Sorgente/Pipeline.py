import os
import pickle

import config as cfg
from Sorgente.Preprocessing.preprocesing        import Preprocessore
from Sorgente.Espansione.espansione_query          import EspansoreQuery
from Sorgente.Indicizzazione.indice_invertito     import IndiceInvertito
from Sorgente.Recupero.metadati                   import GestoreMetadati
from Sorgente.Recupero.bm25                       import CalcolatoreRanking
from Sorgente.Classificazione.classificatore_nli  import ClassificatoreNLI
from Sorgente.Recupero.reranker import Reranker
from Sorgente.Valutazione.metriche_ir             import MetricheIR
from Sorgente.Valutazione.metriche_nlp            import MetricheNLP


class Pipeline:

    def __init__(self, api_key: str):
        print('Inizializzazione pipeline...')
        self._preprocessore = Preprocessore()
        self._espansore     = EspansoreQuery(self._preprocessore)
        print('  ✓ Preprocessore e espansore query')
        self._corpus = self._carica_corpus()
        self._indice = self._carica_indice()
        print(f'  ✓ Corpus ({len(self._corpus):,} documenti) e indice '
              f'({self._indice.numero_termini:,} termini)')
        self._metadati = GestoreMetadati(self._corpus)
        self._ranking  = CalcolatoreRanking(
            self._indice,
            self._preprocessore,
            self._metadati
        )
        print('  ✓ Metadati e ranking BM25 esteso')
        self._classificatore = ClassificatoreNLI(api_key)
        self._reranker = Reranker() if cfg.RERANKING_ATTIVO else None
        print('  ✓ Reranker neurale' if cfg.RERANKING_ATTIVO else '  - Reranker disabilitato')
        print('  ✓ Classificatore NLI')
        self._metriche_ir  = MetricheIR()
        self._metriche_nlp = MetricheNLP()
        print('  ✓ Moduli metriche')
        print('Pipeline pronta.\n')

    def _carica_corpus(self) -> dict:
        if cfg.SORGENTE_CORPUS == 'fever':
            if not os.path.exists(cfg.PERCORSO_CORPUS_ARRICCHITO):
                raise FileNotFoundError(
                    f"Corpus non trovato: {cfg.PERCORSO_CORPUS_ARRICCHITO}\n"
                    "Esegui prima il notebook 00_studio_dataset.ipynb"
                )
            with open(cfg.PERCORSO_CORPUS_ARRICCHITO, 'rb') as f:
                return pickle.load(f)
        else:
            return {}

    def _carica_indice(self) -> IndiceInvertito:
        indice = IndiceInvertito()
        if cfg.SORGENTE_CORPUS == 'fever':
            if not os.path.exists(cfg.PERCORSO_INDICE):
                raise FileNotFoundError(
                    f"Indice non trovato: {cfg.PERCORSO_INDICE}\n"
                    "Esegui prima il notebook 02_test_indice.ipynb"
                )
            indice.carica(cfg.PERCORSO_INDICE)
        return indice

    def verifica(
            self,
            claim: str,
            sentence_ids_map: dict | None = None
    ) -> dict:
        token_query = self._espansore.espandi(claim)

        if cfg.SORGENTE_CORPUS == 'web':
            risultati_bm25, corpus_da_usare = self._recupera_da_web(claim)
        else:
            # Se reranking attivo recupera più candidati (es. 20)
            # così il cross-encoder ha più documenti tra cui scegliere
            n_candidati = cfg.CANDIDATI_RERANKING if cfg.RERANKING_ATTIVO else cfg.DOCUMENTI_DA_RECUPERARE
            risultati_bm25 = self._ranking.recupera_top_k(
                claim, k=n_candidati, token_query=token_query  # ← passa i token espansi
            )
            corpus_da_usare = self._corpus

        if not risultati_bm25:
            return {
                'claim': claim,
                'verdetto': 'NOT ENOUGH INFO',
                'confidenza': 0.0,
                'pesi_label': {},
                'spiegazione': '',
                'documenti': [],
                'token_query': token_query
            }

        # Re-ranking neurale — riordina i candidati BM25 semanticamente
        # e taglia ai top-K finali da passare all'NLI
        if self._reranker and cfg.RERANKING_ATTIVO:
            risultati_bm25 = self._reranker.reordina(
                claim=claim,
                documenti=risultati_bm25,
                corpus=corpus_da_usare,
                top_k=cfg.DOCUMENTI_DA_RECUPERARE
            )

        risultato_nli = self._classificatore.verifica(
            claim=claim,
            risultati_bm25=risultati_bm25,
            corpus_arricchito=corpus_da_usare,
            sentence_ids_map=sentence_ids_map
        )

        return {
            'claim': claim,
            'verdetto': risultato_nli['verdetto'],
            'confidenza': risultato_nli['confidenza'],
            'pesi_label': risultato_nli['pesi_label'],
            'spiegazione': risultato_nli.get('spiegazione', ''),
            'documenti': risultato_nli['dettagli'],
            'token_query': token_query
        }

    def valuta(
        self,
        claim_con_ground_truth: list[dict],
        k_ir:                   int = cfg.DOCUMENTI_DA_RECUPERARE
    ) -> dict:
        risultati_ir  = []
        label_vere    = []
        label_predette = []
        print(f'Valutazione su {len(claim_con_ground_truth)} claim...')
        for i, item in enumerate(claim_con_ground_truth):
            claim        = item['claim']
            label_vera   = item['label']
            doc_rilevanti = set(item.get('url_documenti', []))
            sids         = item.get('sentence_ids', [])
            sentence_ids_map = {}
            for url, sid in zip(item.get('url_documenti', []), sids):
                if url not in sentence_ids_map:
                    sentence_ids_map[url] = []
                sentence_ids_map[url].append(sid)
            risultati_bm25 = self._ranking.recupera_top_k(claim, k=k_ir)
            id_recuperati  = [r['id_doc'] for r in risultati_bm25]
            if doc_rilevanti:
                risultati_ir.append({
                    'recuperati': id_recuperati,
                    'rilevanti':  doc_rilevanti
                })
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
            if (i + 1) % 10 == 0:
                print(f'  {i+1}/{len(claim_con_ground_truth)} claim elaborate...')
        metriche_ir  = self._metriche_ir.valuta_sistema(risultati_ir)
        metriche_nlp = self._metriche_nlp.valuta_sistema(label_vere, label_predette)
        return {
            'metriche_ir':  metriche_ir,
            'metriche_nlp': metriche_nlp,
            'n_claim':      len(claim_con_ground_truth),
            'n_claim_ir':   len(risultati_ir)
        }

    def _recupera_da_web(self, claim: str) -> tuple[list[dict], dict]:
        from Sorgente.Corpus.Corpus_web import CorpusWeb
        corpus_web     = CorpusWeb(self._preprocessore)
        corpus_runtime = corpus_web.ottieni_documenti(claim)
        if not corpus_runtime:
            return [], {}
        from Sorgente.Indicizzazione.indice_invertito import IndiceInvertito
        indice_temp = IndiceInvertito()
        for id_doc, doc in corpus_runtime.items():
            token = self._preprocessore.elabora(doc.get('text', ''))
            indice_temp.aggiungi_documento(id_doc, token)
        metadati_temp = GestoreMetadati(corpus_runtime)
        ranking_temp = CalcolatoreRanking(
            indice_temp,
            self._preprocessore,
            metadati_temp
        )
        risultati = ranking_temp.recupera_top_k(
            claim, k=cfg.DOCUMENTI_DA_RECUPERARE
        )
        return risultati, corpus_runtime


    def info(self) -> None:
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