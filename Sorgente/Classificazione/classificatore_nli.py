"""
classificatore_nli.py
════════════════════════════════════════════════════════════════════
Classificazione NLI tramite LLM e aggregazione dei verdetti.

Responsabilità:
    1. Chiamare l'API LLM per classificare ogni coppia (claim, evidenza)
    2. Aggregare i verdetti di K documenti in un verdetto finale
    3. Estrarre il testo rilevante dai documenti del corpus

Approccio NLI tramite LLM:
    Invece di un modello NLI dedicato (es. RoBERTa fine-tuned su FEVER),
    usiamo un LLM generale tramite prompt strutturato.
    Vantaggi: zero training, flessibile, buone performance zero-shot.
    Svantaggi: costo API, latenza, uso parziale di conoscenza esterna.

Aggregazione — maggioranza pesata:
    Ogni documento recuperato da BM25 produce un verdetto.
    Il verdetto finale è quello con il maggior peso aggregato,
    dove il peso di ogni verdetto è proporzionale al punteggio
    BM25 del documento che lo ha prodotto.
    Questo garantisce che i documenti più rilevanti influenzino
    di più il verdetto finale rispetto a quelli recuperati
    con bassa rilevanza.

Estrazione evidenza — frasi rilevanti invece di testo troncato:
    Il corpus FEVER indica per ogni claim le frasi specifiche
    (sentence_id) che costituiscono l'evidenza.
    Quando disponibili, usiamo quelle frasi invece del testo
    completo troncato — più preciso e meno token API consumati.
    Per le claim NOT ENOUGH INFO e per la modalità web,
    usiamo il testo completo troncato come fallback.

Limite noto:
    L'LLM può usare conoscenza esterna nonostante l'istruzione
    nel prompt. Questo causa errori su casi limite (es. date
    implicite, inferenze che richiedono calcoli).
    Da citare nella relazione come limite del sistema.

Utilizzo:
    from sorgente.classificazione.classificatore_nli import ClassificatoreNLI

    classificatore = ClassificatoreNLI(api_key='...')
    risultato = classificatore.verifica(
        claim             = 'Obama was born in Hawaii',
        risultati_bm25    = [...],   # output di CalcolatoreRanking
        corpus_arricchito = {...}
    )
════════════════════════════════════════════════════════════════════
"""

from openai import OpenAI

from Sorgente.Classificazione.costruttore_prompt import CostruttorePrompt
import config as cfg


# Label valide — esattamente le stesse del dataset FEVER
LABEL_VALIDE = frozenset({'SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO'})

# Fallback restituito quando la classificazione fallisce o
# l'LLM produce output non riconoscibile.
# NOT ENOUGH INFO è il fallback più sicuro: è meglio non
# pronunciarsi che dare una label errata con sicurezza.
LABEL_FALLBACK = 'NOT ENOUGH INFO'


class ClassificatoreNLI:
    """
    Classifica la relazione tra una claim e i documenti recuperati.

    Gestisce tre responsabilità distinte:
    1. Estrazione del testo rilevante dal corpus
    2. Classificazione di ogni coppia (claim, evidenza) tramite LLM
    3. Aggregazione dei verdetti in un verdetto finale

    Attributes:
        _client:       client OpenAI per le chiamate API
        _costruttore:  costruttore dei prompt
        _modello:      nome del modello LLM da usare
        _temperatura:  temperatura per la generazione (0 = deterministico)
        _max_token:    numero massimo di token nella risposta
        _k_documenti:  numero di documenti da classificare per query
        _max_caratteri_evidenza: lunghezza massima del testo evidenza
    """

    def __init__(self, api_key: str):
        """
        Inizializza il classificatore con le credenziali API.

        I parametri di configurazione vengono letti da configurazione.py
        per centralizzare tutte le impostazioni del sistema.

        Args:
            api_key: chiave API OpenAI
        """
        self._client      = OpenAI(api_key=api_key)
        self._costruttore = CostruttorePrompt()
        self._modello     = cfg.MODELLO_LLM
        self._temperatura = cfg.TEMPERATURA
        self._max_token   = cfg.TOKEN_MASSIMI
        self._k_documenti = cfg.DOCUMENTI_DA_RECUPERARE
        self._max_caratteri_evidenza = 1000

    # ── Estrazione evidenza ───────────────────────────────────────

    def _estrai_frase_da_lines(self, lines: str, sentence_id: int) -> str:
        """
        Estrae una frase specifica dal campo 'lines' di un documento.

        Il campo 'lines' ha formato: '0\\tFrase zero.\\n1\\tFrase uno.'
        Usiamo il sentence_id indicato da FEVER come evidenza ground truth
        invece del testo completo — più preciso e meno token consumati.

        Args:
            lines:       campo 'lines' del documento corpus
            sentence_id: indice della frase da estrarre

        Returns:
            Testo della frase, stringa vuota se non trovata
        """
        if not lines:
            return ''
        for riga in lines.split('\n'):
            riga = riga.strip()
            if not riga:
                continue
            parti = riga.split('\t', 1)
            if len(parti) == 2:
                try:
                    if int(parti[0]) == sentence_id:
                        return parti[1].strip()
                except ValueError:
                    continue
        return ''

    def _prepara_evidenza(
        self,
        id_doc:            str,
        corpus_arricchito: dict,
        sentence_ids:      list[int] | None = None
    ) -> str:
        """
        Prepara il testo dell'evidenza per un documento.

        Strategia in ordine di preferenza:
        1. Se sentence_ids è fornito → usa le frasi specifiche indicate
           da FEVER come evidenza (più preciso, meno token)
        2. Altrimenti → usa il testo completo troncato (fallback)

        Il testo viene sempre troncato a _max_caratteri_evidenza
        per contenere i costi e il numero di token in input all'LLM.

        Args:
            id_doc:            identificatore del documento
            corpus_arricchito: corpus con testo dei documenti
            sentence_ids:      indici delle frasi rilevanti (opzionale)

        Returns:
            Testo dell'evidenza pronto per il prompt
        """
        doc = corpus_arricchito.get(id_doc, {})
        if not doc:
            return ''

        # Priorità 1: frasi specifiche indicate da FEVER
        if sentence_ids:
            lines = doc.get('lines', '')
            frasi = []
            for sid in sentence_ids:
                frase = self._estrai_frase_da_lines(lines, sid)
                if frase:
                    frasi.append(frase)
            if frasi:
                return ' '.join(frasi)[:self._max_caratteri_evidenza]

        # Fallback: testo completo troncato
        testo = doc.get('text', '') or ''
        return testo[:self._max_caratteri_evidenza]

    # ── Classificazione singola ───────────────────────────────────

    def _classifica_coppia(self, claim: str, evidenza: str) -> str:
        """
        Classifica la relazione tra una claim e un'evidenza.

        Invia il prompt all'LLM e analizza la risposta.

        Parametri LLM scelti:
        - temperatura=0: output deterministico e riproducibile.
          Stesso input → stessa risposta. Fondamentale per la
          valutazione scientifica.
        - max_tokens=10: la risposta deve essere una sola label
          (max 3 parole). Riduce costi e latenza.

        Gestione risposta:
        - Verifica esatta della label (case insensitive dopo upper())
        - Gestione parziale: rimuove punto finale se presente
        - Fallback: NOT ENOUGH INFO se risposta non riconoscibile

        Args:
            claim:    testo della claim
            evidenza: testo del documento (premessa)

        Returns:
            Una delle tre label valide
        """
        if not evidenza or not evidenza.strip():
            return LABEL_FALLBACK

        try:
            messaggi = self._costruttore.prepara_messaggi(claim, evidenza)
            risposta = self._client.chat.completions.create(
                model       = self._modello,
                messages    = messaggi,
                temperature = self._temperatura,
                max_tokens  = self._max_token
            )
            testo = risposta.choices[0].message.content.strip().upper()

            # Verifica esatta
            if testo in LABEL_VALIDE:
                return testo

            # Gestione punto finale spurio (es. "SUPPORTS.")
            testo_pulito = testo.rstrip('.')
            if testo_pulito in LABEL_VALIDE:
                return testo_pulito

            # Risposta non riconoscibile → fallback
            print(f'  ⚠️  Risposta LLM non riconosciuta: "{testo}" → {LABEL_FALLBACK}')
            return LABEL_FALLBACK

        except Exception as e:
            print(f'  ❌ Errore API OpenAI: {e}')
            return LABEL_FALLBACK

    # ── Aggregazione ──────────────────────────────────────────────

    def _aggrega_verdetti(
        self,
        verdetti: list[str],
        punteggi: list[float]
    ) -> tuple[str, dict]:
        """
        Aggrega i verdetti di K documenti in un verdetto finale.

        Usa la maggioranza pesata: somma i punteggi BM25 per ogni
        label e sceglie quella con il punteggio aggregato maggiore.

        Perché pesare per BM25:
            Un documento con punteggio BM25 alto è più rilevante
            per la claim. Il suo verdetto deve influenzare di più
            il risultato finale rispetto a documenti recuperati
            con bassa rilevanza testuale.

        Perché non la maggioranza semplice:
            Con k=5 documenti e ranking imperfetto, i documenti
            meno rilevanti (rank 4-5) potrebbero essere rumore.
            Il peso BM25 li penalizza naturalmente.

        Args:
            verdetti: lista di label per ogni documento
            punteggi: lista di punteggi BM25 corrispondenti

        Returns:
            Tupla (verdetto_finale, pesi_per_label):
                verdetto_finale: label con peso aggregato massimo
                pesi_per_label:  {'SUPPORTS': x, 'REFUTES': y, 'NOT ENOUGH INFO': z}
        """
        if not verdetti:
            return LABEL_FALLBACK, {}

        pesi: dict[str, float] = {
            'SUPPORTS':        0.0,
            'REFUTES':         0.0,
            'NOT ENOUGH INFO': 0.0
        }
        for verdetto, punteggio in zip(verdetti, punteggi):
            if verdetto in pesi:
                pesi[verdetto] += punteggio

        verdetto_finale = max(pesi, key=lambda k: pesi[k])
        return verdetto_finale, pesi

    # ── Interfaccia pubblica ──────────────────────────────────────

    def verifica(
        self,
        claim:             str,
        risultati_bm25:    list[dict],
        corpus_arricchito: dict,
        sentence_ids_map:  dict[str, list[int]] | None = None
    ) -> dict:
        """
        Verifica una claim attraverso la pipeline NLI completa.

        Passi:
            1. Per ogni documento recuperato da BM25:
               a. Estrae il testo dell'evidenza (frasi specifiche o testo completo)
               b. Classifica la coppia (claim, evidenza) con l'LLM
            2. Aggrega i verdetti con maggioranza pesata
            3. Restituisce il risultato completo con tutti i dettagli

        Args:
            claim:             testo della claim da verificare
            risultati_bm25:    output di CalcolatoreRanking.recupera_top_k()
            corpus_arricchito: corpus con testo dei documenti
            sentence_ids_map:  {id_doc: [sentence_ids]} per le frasi rilevanti
                               Opzionale — se None usa il testo completo

        Returns:
            Dizionario con:
                'claim':        testo originale della claim
                'verdetto':     label finale aggregata
                'pesi_label':   pesi aggregati per label
                'confidenza':   percentuale del peso della label vincente
                'dettagli':     lista dettagli per ogni documento classificato
                    - 'id_doc':    identificatore documento
                    - 'titolo':    titolo leggibile
                    - 'punteggio': punteggio BM25 esteso
                    - 'verdetto':  label assegnata dall'LLM
                    - 'evidenza':  testo usato come evidenza (troncato)
        """
        if not risultati_bm25:
            return {
                'claim':      claim,
                'verdetto':   LABEL_FALLBACK,
                'pesi_label': {},
                'confidenza': 0.0,
                'dettagli':   []
            }

        verdetti = []
        punteggi = []
        dettagli = []

        for risultato in risultati_bm25:
            id_doc = risultato['id_doc']

            # Estrae le frasi rilevanti se disponibili
            sids = sentence_ids_map.get(id_doc) if sentence_ids_map else None
            evidenza = self._prepara_evidenza(id_doc, corpus_arricchito, sids)

            # Classificazione LLM
            label = self._classifica_coppia(claim, evidenza)

            verdetti.append(label)
            punteggi.append(risultato['punteggio_finale'])
            dettagli.append({
                'id_doc':    id_doc,
                'titolo':    risultato['titolo'],
                'punteggio': risultato['punteggio_finale'],
                'verdetto':  label,
                'evidenza':  evidenza[:200]   # troncato per la visualizzazione
            })

        # Aggregazione
        verdetto_finale, pesi = self._aggrega_verdetti(verdetti, punteggi)

        # Confidenza: percentuale del peso della label vincente sul totale
        peso_totale = sum(pesi.values())
        confidenza  = (pesi[verdetto_finale] / peso_totale * 100
                       if peso_totale > 0 else 0.0)

        return {
            'claim':      claim,
            'verdetto':   verdetto_finale,
            'pesi_label': {k: round(v, 4) for k, v in pesi.items()},
            'confidenza': round(confidenza, 1),
            'dettagli':   dettagli
        }