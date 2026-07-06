"""
corpus_fever.py
════════════════════════════════════════════════════════════════════
Sorgente di documenti dal corpus FEVER pre-costruito.

Responsabilità:
    Caricare e fornire i documenti Wikipedia del corpus FEVER
    arricchito con metadati (numero_link_entrata, timestamp).

Differenza con corpus_web.py:
    CorpusFever lavora su un corpus fisso pre-costruito e salvato
    su disco nel notebook 00. I documenti sono sempre gli stessi
    e l'indice BM25 viene costruito una volta sola (notebook 02).
    Questo permette la valutazione rigorosa con metriche oggettive.

    CorpusWeb invece recupera documenti freschi da Wikipedia
    ad ogni query — nessun indice pre-costruito, ma risponde
    a qualsiasi domanda anche fuori dal dominio FEVER.

Utilizzo:
    from sorgente.corpus.corpus_fever import CorpusFever

    corpus = CorpusFever('corpus_fever_arricchito.pkl')
    documenti = corpus.ottieni_documenti('Obama was born in Hawaii')
    documento = corpus.ottieni_documento('Barack_Obama')
════════════════════════════════════════════════════════════════════
"""

import pickle
from Sorgente.Corpus.corpus_base import CorpusBase


class CorpusFever(CorpusBase):
    """
    Sorgente di documenti dal corpus FEVER arricchito.

    Carica il corpus una volta sola in memoria all'inizializzazione.
    Tutti i metodi successivi sono O(1) — semplici accessi al dizionario.

    Il corpus contiene ~14.500 pagine Wikipedia citate nelle
    evidenze del dataset FEVER, arricchite con numero_link_entrata
    e timestamp generato nel notebook 00.

    Attributes:
        _corpus: dizionario {id_doc: {lines, text, numero_link_entrata, timestamp}}
    """

    def __init__(self, percorso_corpus: str):
        """
        Carica il corpus arricchito da disco.

        Args:
            percorso_corpus: percorso del file corpus_fever_arricchito.pkl

        Raises:
            FileNotFoundError: se il file non esiste
        """
        with open(percorso_corpus, 'rb') as f:
            self._corpus: dict = pickle.load(f)

        # Rimuove eventuali documenti con id vuoto (artefatto del corpus FEVER)
        self._corpus = {k: v for k, v in self._corpus.items() if k.strip()}

    # ── Implementazione interfaccia CorpusBase ────────────────────

    def ottieni_documenti(self, query: str = '') -> dict:
        """
        Restituisce l'intero corpus FEVER.

        In CorpusFever la query viene ignorata: il corpus è fisso
        e l'indice BM25 si occupa del recupero per rilevanza.
        Questo metodo serve principalmente per compatibilità
        con l'interfaccia CorpusBase e per la modalità web.

        Args:
            query: ignorato in questa implementazione

        Returns:
            Dizionario completo {id_doc: documento}
        """
        return self._corpus

    def ottieni_documento(self, id_doc: str) -> dict:
        """
        Restituisce un singolo documento dato il suo id.

        Args:
            id_doc: titolo della pagina Wikipedia (es. 'Barack_Obama')

        Returns:
            Dizionario del documento, vuoto se non trovato
        """
        return self._corpus.get(id_doc, {})

    def numero_documenti(self) -> int:
        """
        Restituisce il numero di pagine nel corpus FEVER.

        Returns:
            Numero intero (~14.500 per il corpus FEVER standard)
        """
        return len(self._corpus)

    # ── Metodi specifici FEVER ────────────────────────────────────

    def estrai_frase(self, id_doc: str, sentence_id: int) -> str:
        """
        Estrae una frase specifica da un documento.

        Usa il campo 'lines' nel formato FEVER:
            '0\\tFrase zero.\\n1\\tFrase uno.'

        Questo metodo è specifico di CorpusFever perché solo
        il corpus FEVER ha il campo 'lines' strutturato con
        i sentence_id. Il corpus web non lo ha.

        Args:
            id_doc:      identificatore del documento
            sentence_id: indice della frase da estrarre

        Returns:
            Testo della frase, stringa vuota se non trovata
        """
        doc   = self._corpus.get(id_doc, {})
        lines = doc.get('lines', '')
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

    def costruisci_sentence_ids_map(
        self,
        url_documenti: list[str],
        sentence_ids:  list[int]
    ) -> dict:
        """
        Costruisce la mappa {id_doc: [sentence_ids]} per il classificatore.

        Prende le liste parallele url_documenti e sentence_ids
        (come compaiono nel dataset FEVER) e le trasforma nella
        struttura attesa da ClassificatoreNLI.verifica().

        Args:
            url_documenti: lista di titoli pagina Wikipedia
            sentence_ids:  lista di indici frase corrispondenti

        Returns:
            {id_doc: [lista di sentence_id per quel documento]}
        """
        mappa: dict[str, list[int]] = {}
        for url, sid in zip(url_documenti, sentence_ids):
            if url:
                if url not in mappa:
                    mappa[url] = []
                mappa[url].append(sid)
        return mappa