import math

from openai import OpenAI
from Sorgente.Classificazione.costruttore_prompt import CostruttorePrompt
import config as cfg
from Sorgente.Preprocessing.preprocesing import Preprocessore

LABEL_VALIDE = frozenset({'SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO'})
LABEL_FALLBACK = 'NOT ENOUGH INFO'


class ClassificatoreNLI:

    def __init__(self, api_key: str):
        self._client      = OpenAI(api_key=api_key)
        self._costruttore = CostruttorePrompt()
        self._modello     = cfg.MODELLO_LLM
        self._temperatura = cfg.TEMPERATURA
        self._max_token   = cfg.TOKEN_MASSIMI
        self._k_documenti = cfg.DOCUMENTI_DA_RECUPERARE
        self._max_caratteri_evidenza = 1000

    def _estrai_frase_da_lines(self, lines: str, sentence_id: int) -> str:
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
            id_doc: str,
            corpus: dict,
            sentence_ids: list[int] | None = None,
            token_query: list[str] | None = None
    ) -> str:
        print(f"DEBUG _prepara_evidenza: id_doc={id_doc}, token_query={token_query}")

        """
        Estrae il testo dell'evidenza per un documento.

        Priorità:
        1. Frasi specifiche indicate da FEVER (sentence_ids)
        2. Frasi contenenti token della query (se token_query forniti)
        3. Testo completo troncato (fallback)
        """
        doc = corpus.get(id_doc, {})
        if not doc:
            return ''

        # Priorità 1: frasi specifiche FEVER
        if sentence_ids:
            lines = doc.get('lines', '')
            frasi = []
            for sid in sentence_ids:
                frase = self._estrai_frase_da_lines(lines, sid)
                if frase:
                    frasi.append(frase)
            if frasi:
                return ' '.join(frasi)[:self._max_caratteri_evidenza]

            # Priorità 2: frasi rilevanti per token query
        if token_query:
            lines = doc.get('lines', '') or ''
            frasi_rilevanti = []
            righe_valide = []

            for riga in lines.split('\n'):
                parti = riga.split('\t', 1)
                if len(parti) == 2:
                    testo_frase = parti[1].strip()
                    if not testo_frase:
                        continue
                    righe_valide.append(testo_frase)
                    testo_lower = testo_frase.lower()
                    n_match = sum(1 for token in token_query if token in testo_lower)
                    if n_match > 0:
                        # Salva con numero di match per ordinare dopo
                        frasi_rilevanti.append((n_match, testo_frase))

            if frasi_rilevanti:
                # Ordina per numero di match decrescente — le frasi più rilevanti prima
                frasi_rilevanti.sort(key=lambda x: -x[0])
                testi_ordinati = [f for _, f in frasi_rilevanti]
                return ' '.join(testi_ordinati)[:self._max_caratteri_evidenza]

            if righe_valide:
                campione = righe_valide[:5] + righe_valide[-5:]
                return ' '.join(campione)[:self._max_caratteri_evidenza]

        # Fallback finale: testo completo troncato
        return (doc.get('text', '') or '')[:self._max_caratteri_evidenza]

    def _classifica_coppia(self, claim: str, evidenza: str) -> str:
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

    def _aggrega_verdetti(
            self,
            verdetti: list[str],
            punteggi: list[float]
    ) -> tuple[str, dict, float]:
        """
        Aggrega i verdetti combinando segnale semantico e consenso tra documenti.

        Strategia a due componenti:

        1. SCORE SEMANTICO (dal cross-encoder, già normalizzato [0,1] via sigmoide)
           Per ogni label prende il punteggio massimo tra i documenti che l'hanno votata.
           Il massimo è più robusto della media: un documento molto rilevante con score
           alto è più informativo di tanti documenti mediocri.

        2. CONSENSO (frazione di documenti che votano per la label vincente)
           Misura quanto i documenti sono d'accordo tra loro.
           Es: 4/5 documenti SUPPORTS → consenso 0.80

        CONFIDENZA FINALE = score_semantico * 0.6 + consenso * 0.4
           Il peso maggiore va allo score semantico perché viene dal cross-encoder,
           che ha già valutato la coppia (claim, documento) — è il segnale più ricco.
           Il consenso corregge i casi in cui un singolo documento ha score alto
           ma tutti gli altri dicono il contrario.

        NOT ENOUGH INFO viene penalizzata (×0.5) perché è una label debole:
        significa solo che quel documento non contiene l'informazione,
        non che la claim sia falsa o vera.
        """
        if not verdetti:
            return LABEL_FALLBACK, {}, 0.0

        LABELS = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']

        # 1. Score semantico massimo per ogni label
        score_max: dict[str, float] = {l: 0.0 for l in LABELS}
        conteggi:  dict[str, int]   = {l: 0   for l in LABELS}

        for verdetto, punteggio in zip(verdetti, punteggi):
            if verdetto in score_max:
                conteggi[verdetto] += 1
                if punteggio > score_max[verdetto]:
                    score_max[verdetto] = punteggio

        # I punteggi passati sono già sigmoide*100 (range 0-100) se viene
        # dal reranker, oppure punteggio_finale BM25 (range variabile) se no.
        # Normalizziamo entrambi in [0,1] dividendo per il massimo osservato.
        max_score_globale = max(score_max.values()) or 1.0
        score_norm: dict[str, float] = {
            l: score_max[l] / max_score_globale for l in LABELS
        }

        # 2. Consenso: frazione documenti per label
        n_totale = len(verdetti)
        consenso: dict[str, float] = {
            l: conteggi[l] / n_totale for l in LABELS
        }

        # 3. Score combinato con penalità NEI
        PENALITA_NEI = 0.5
        PESO_SEMANTICO = 0.6
        PESO_CONSENSO  = 0.4

        score_combinato: dict[str, float] = {}
        for l in LABELS:
            s = score_norm[l] * PESO_SEMANTICO + consenso[l] * PESO_CONSENSO
            if l == 'NOT ENOUGH INFO':
                s *= PENALITA_NEI
            score_combinato[l] = s

        verdetto_finale = max(score_combinato, key=lambda k: score_combinato[k])

        # Confidenza: score combinato del vincitore normalizzato sul totale
        # Rappresenta quanto il verdetto domina rispetto alle alternative
        totale_score = sum(score_combinato.values()) or 1.0
        confidenza = round(score_combinato[verdetto_finale] / totale_score * 100, 1)

        # pesi_label esposti verso l'esterno: score semantico grezzo per label
        # (usato dalla UI per le barre di distribuzione)
        pesi_esposti = {l: round(score_combinato[l], 4) for l in LABELS}

        return verdetto_finale, pesi_esposti, confidenza

    def _genera_spiegazione(
            self,
            claim: str,
            verdetto: str,
            evidenze: list[str]
    ) -> str:
        testo_evidenze = '\n'.join(
            f"- {ev[:300]}" for ev in evidenze[:3] if ev
        )
        if not testo_evidenze:
            return ''

        istruzione = {
            'SUPPORTS': 'explain what specific information in the evidence confirms the claim.',
            'REFUTES': 'explain what specific information in the evidence contradicts the claim.',
            'NOT ENOUGH INFO': 'explain why the evidence does not contain enough information to verify the claim.'
        }.get(verdetto, 'explain the verdict.')

        prompt = f"""You are a fact-checking assistant. Based ONLY on the evidence below, {istruzione}

    CLAIM: {claim}
    VERDICT: {verdetto}

    EVIDENCE:
    {testo_evidenze}

    Write 1-2 clear sentences. Be specific about what the evidence says or does not say. Do not use external knowledge."""

        try:
            risposta = self._client.chat.completions.create(
                model=self._modello,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.3,
                max_tokens=100
            )
            return risposta.choices[0].message.content.strip()
        except Exception:
            return ''

    def verifica(
            self,
            claim: str,
            risultati_bm25: list[dict],
            corpus_arricchito: dict,
            sentence_ids_map: dict | None = None
    ) -> dict:
        if not risultati_bm25:
            return {
                'claim': claim,
                'verdetto': LABEL_FALLBACK,
                'pesi_label': {},
                'confidenza': 0.0,
                'dettagli': []
            }

        # Calcola i token della query una volta sola
        # Importa il preprocessore solo per estrarre i token

        _prep = Preprocessore()
        token_query = _prep.elabora(claim)
        print(f"DEBUG token_query: {token_query}")  # aggiungi temporaneamente

        verdetti = []
        punteggi = []
        dettagli = []

        for risultato in risultati_bm25:
            id_doc = risultato['id_doc']
            sids = sentence_ids_map.get(id_doc) if sentence_ids_map else None

            # Passa i token query per l'estrazione frasi rilevanti
            evidenza = self._prepara_evidenza(
                id_doc, corpus_arricchito, sids, token_query
            )
            print(f"DEBUG evidenza per {id_doc}: {evidenza[:100]}")  # aggiungi temporaneamente

            label = self._classifica_coppia(claim, evidenza)
            verdetti.append(label)
            if 'score_reranking' in risultato:
                # Applica la sigmoide per portare il logit in un range 0-1
                score_norm = 1 / (1 + math.exp(-risultato['score_reranking']))
                # Moltiplica per allinearlo a una scala comparabile
                punteggi.append(score_norm * 100)
            else:
                punteggi.append(risultato['punteggio_finale'])

            # Crea il dizionario base
            doc_dettaglio = {
                'id_doc': id_doc,
                'titolo': risultato['titolo'],
                'punteggio': risultato['punteggio_finale'],
                'verdetto': label,
                'evidenza': evidenza[:500]
            }

            # Se il reranking è attivo, recupera e propaga la chiave
            if 'score_reranking' in risultato:
                doc_dettaglio['score_reranking'] = risultato['score_reranking']

            dettagli.append(doc_dettaglio)

        verdetto_finale, pesi, confidenza = self._aggrega_verdetti(verdetti, punteggi)

        evidenze_testo = [d['evidenza'] for d in dettagli if d['evidenza']]
        spiegazione = self._genera_spiegazione(claim, verdetto_finale, evidenze_testo)

        return {
            'claim': claim,
            'verdetto': verdetto_finale,
            'pesi_label': {k: round(v, 4) for k, v in pesi.items()},
            'confidenza': confidenza,
            'spiegazione': spiegazione,
            'dettagli': dettagli
        }