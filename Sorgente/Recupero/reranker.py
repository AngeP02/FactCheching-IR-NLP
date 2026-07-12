
"""
reranker.py
════════════════════════════════════════════════════════════════════
Re-ranking neurale dei documenti recuperati da BM25.

Responsabilità:
    Riordinare i documenti recuperati da BM25 usando un cross-encoder
    che valuta la rilevanza semantica della coppia (claim, documento)
    invece della semplice sovrapposizione lessicale.

Perché il re-ranking:
    BM25 è efficace per il recall (trova i documenti candidati) ma
    soffre di topic drift: documenti topicamente correlati ma
    informativamente irrilevanti ricevono score alto per la presenza
    frequente dei termini della query.

    Il cross-encoder risolve questo limite valutando la coppia
    (claim, documento) nel suo insieme, catturando la rilevanza
    semantica specifica invece della semplice frequenza dei termini.

    Esempio:
        Query: "Obama born in Hawaii"
        BM25 rank 1: "Presidency of Barack Obama" (molte occorrenze
                      di "Obama" → score alto ma non risponde alla query)
        Cross-encoder rank 1: "Barack Obama" (semanticamente più
                               rilevante per il luogo di nascita)

Architettura a due stadi:
    Stadio 1 — BM25:         recupera top-N con alto recall (veloce)
    Stadio 2 — Cross-encoder: riordina top-N con alta precision (lento)

    N è più grande di K (es. N=20, K=5) per dare al cross-encoder
    abbastanza candidati tra cui scegliere senza perdere recall.

Modello usato:
    cross-encoder/ms-marco-MiniLM-L-6-v2
    - Addestrato su MS MARCO (dataset di question answering)
    - 6 layer transformer — bilanciamento velocità/qualità
    - Input max: 512 token
    - Output: score di rilevanza (non normalizzato)

Utilizzo:
    from sorgente.recupero.reranker import Reranker

    reranker  = Reranker()
    riordinati = reranker.reordina(
        claim      = 'Obama was born in Hawaii',
        documenti  = risultati_bm25,   # lista da CalcolatoreRanking
        corpus     = corpus_arricchito,
        top_k      = 5
    )
════════════════════════════════════════════════════════════════════
"""

from sentence_transformers import CrossEncoder
import config as cfg


class Reranker:
    """
    Re-ranking neurale dei documenti recuperati da BM25.

    Usa un cross-encoder pre-addestrato per riordinare i documenti
    in base alla rilevanza semantica rispetto alla claim.

    Attributes:
        _modello:         cross-encoder pre-addestrato
        _max_caratteri:   lunghezza massima del testo documento
    """

    NOME_MODELLO = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

    def __init__(self):
        """
        Carica il cross-encoder in memoria.

        Il modello viene scaricato automaticamente da HuggingFace
        alla prima esecuzione (~80MB) e messo in cache localmente.
        """
        print(f'  Caricamento cross-encoder: {self.NOME_MODELLO}...')
        self._modello = CrossEncoder(self.NOME_MODELLO)
        self._max_caratteri = 512   # limite token del modello
        print('  ✓ Cross-encoder caricato')

    # ── Metodi privati ────────────────────────────────────────────

    def _estrai_testo_documento(
        self,
        id_doc: str,
        corpus: dict
    ) -> str:
        """
        Estrae il testo del documento per il cross-encoder.

        Usa il testo completo troncato — il cross-encoder lavora
        meglio con testo continuo che con frasi estratte.

        Args:
            id_doc: identificatore del documento
            corpus: corpus arricchito

        Returns:
            Testo del documento troncato a _max_caratteri
        """
        doc = corpus.get(id_doc, {})
        testo = doc.get('text', '') or ''
        return testo[:self._max_caratteri]

    # ── Interfaccia pubblica ──────────────────────────────────────

    def reordina(
        self,
        claim: str,
        documenti: list[dict],
        corpus: dict,
        top_k: int = None
    ) -> list[dict]:
        """
        Riordina i documenti per rilevanza semantica rispetto alla claim.

        Per ogni documento costruisce la coppia (claim, testo_documento)
        e la passa al cross-encoder che produce uno score semantico.
        I documenti vengono poi riordinati per score decrescente.

        Il punteggio BM25 originale viene conservato nel dizionario
        per trasparenza e per la pagina sviluppatori della Streamlit app.

        Args:
            claim:     testo della claim da verificare
            documenti: lista di dizionari da CalcolatoreRanking.recupera_top_k()
            corpus:    corpus arricchito con il testo dei documenti
            top_k:     quanti documenti restituire dopo il re-ranking
                       (default: tutti)

        Returns:
            Lista di dizionari riordinati per score cross-encoder,
            con campo aggiuntivo 'score_reranking'.
            Stessa struttura dell'input — compatibile con il resto
            della pipeline senza modifiche.
        """
        if not documenti:
            return []

        # Costruisce le coppie (claim, testo_documento) per il cross-encoder
        coppie = []
        for doc in documenti:
            testo = self._estrai_testo_documento(doc['id_doc'], corpus)
            coppie.append((claim, testo))

        # Calcola gli score semantici per tutte le coppie in un batch
        # Il batch processing è molto più efficiente delle chiamate singole
        scores = self._modello.predict(coppie)

        # Aggiunge lo score di re-ranking a ogni documento
        documenti_con_score = []
        for doc, score in zip(documenti, scores):
            doc_arricchito = dict(doc)   # copia per non modificare l'originale
            doc_arricchito['score_reranking'] = round(float(score), 4)
            documenti_con_score.append(doc_arricchito)

        # Riordina per score cross-encoder decrescente
        documenti_riordinati = sorted(
            documenti_con_score,
            key=lambda x: -x['score_reranking']
        )

        # Taglia a top_k se specificato
        if top_k:
            return documenti_riordinati[:top_k]
        return documenti_riordinati

    def statistiche_reranking(
        self,
        documenti_prima: list[dict],
        documenti_dopo: list[dict]
    ) -> dict:
        """
        Calcola le statistiche del re-ranking per il debug.

        Confronta l'ordine prima e dopo il re-ranking per capire
        quanto il cross-encoder modifica il ranking di BM25.

        Args:
            documenti_prima: lista prima del re-ranking (ordine BM25)
            documenti_dopo:  lista dopo il re-ranking (ordine cross-encoder)

        Returns:
            Dizionario con statistiche sul cambiamento di ranking
        """
        if not documenti_prima or not documenti_dopo:
            return {}

        ids_prima = [d['id_doc'] for d in documenti_prima]
        ids_dopo = [d['id_doc'] for d in documenti_dopo]

        # Quanti documenti hanno cambiato posizione?
        cambiamenti = sum(
            1 for i, id_doc in enumerate(ids_dopo)
            if i < len(ids_prima) and id_doc != ids_prima[i]
        )

        # Il top-1 è cambiato?
        top1_cambiato = (
            ids_prima[0] != ids_dopo[0]
            if ids_prima and ids_dopo else False
        )

        return {
            'documenti_totali': len(documenti_prima),
            'posizioni_cambiate': cambiamenti,
            'top1_cambiato': top1_cambiato,
            'nuovo_top1': ids_dopo[0] if ids_dopo else None,
            'vecchio_top1': ids_prima[0] if ids_prima else None
        }