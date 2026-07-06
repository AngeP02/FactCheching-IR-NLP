"""
corpus_web.py
════════════════════════════════════════════════════════════════════
Sorgente di documenti da Wikipedia in tempo reale.

Responsabilità:
    Recuperare documenti Wikipedia freschi per una query,
    permettendo al sistema di rispondere a domande fuori
    dal dominio del corpus FEVER.

Differenza con corpus_fever.py:
    CorpusWeb non ha un corpus pre-costruito né un indice BM25
    pre-calcolato. Per ogni query:
    1. Cerca su Wikipedia le pagine più rilevanti
    2. Scarica il testo di ogni pagina
    3. Costruisce un mini-corpus al volo
    4. La pipeline costruisce un mini-indice temporaneo su questi documenti

    Questo permette di rispondere a domande come:
    "Sono vietati i cellulari in classe in Italia?"
    che non sono nel corpus FEVER.

Limiti rispetto a CorpusFever:
    - Non calcolabile con metriche oggettive (no ground truth)
    - Più lento (chiamate API Wikipedia per ogni query)
    - Il testo Wikipedia attuale può differire dalla versione 2017
    - I metadati (numero_link, timestamp) sono simulati

Metadati per modalità web:
    In modalità web non abbiamo il conteggio reale dei link in entrata
    né il timestamp storico. Usiamo:
    - numero_link_entrata: lunghezza del testo come proxy di autorevolezza
    - timestamp: data odierna (tutti i documenti hanno freshness=1.0)
    Questi valori non discriminano tra documenti — authority e freshness
    hanno impatto minimo, il ranking dipende quasi solo da BM25.

Utilizzo:
    from sorgente.corpus.corpus_web import CorpusWeb
    from sorgente.preprocessing.preprocessore import Preprocessore

    preprocessore = Preprocessore()
    corpus = CorpusWeb(preprocessore, max_pagine=5)
    documenti = corpus.ottieni_documenti('cellulari vietati in classe Italia')
════════════════════════════════════════════════════════════════════
"""

from datetime import datetime
import wikipediaapi

from Sorgente.Corpus.corpus_base       import CorpusBase
from Sorgente.Preprocessing.preprocesing import Preprocessore


# Timestamp fisso per tutti i documenti web — data della query.
# In modalità web tutti i documenti hanno lo stesso freshness_score (1.0).
# Non è discriminante ma è necessario per compatibilità con GestoreMetadati.
TIMESTAMP_OGGI = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')


class CorpusWeb(CorpusBase):
    """
    Sorgente di documenti da Wikipedia in tempo reale.

    Per ogni query cerca le pagine Wikipedia più rilevanti,
    ne scarica il testo e costruisce un mini-corpus compatibile
    con la struttura attesa dalla pipeline.

    Attributes:
        _preprocessore: per estrarre termini chiave dalla query
        _max_pagine:    numero massimo di pagine da recuperare
        _wiki:          client Wikipedia API
    """

    def __init__(
        self,
        preprocessore: Preprocessore,
        max_pagine:    int = 5
    ):
        """
        Inizializza il corpus web.

        Args:
            preprocessore: istanza del Preprocessore
            max_pagine:    numero massimo di pagine Wikipedia da recuperare
        """
        self._preprocessore = preprocessore
        self._max_pagine    = max_pagine
        self._wiki          = wikipediaapi.Wikipedia(
            language   = 'en',
            user_agent = 'FactChecker-ProgettoEsame/1.0'
        )

    # ── Implementazione interfaccia CorpusBase ────────────────────

    def ottieni_documenti(self, query: str) -> dict:
        """
        Recupera documenti Wikipedia rilevanti per la query.

        Strategia di ricerca:
        1. Usa la query originale come termine di ricerca Wikipedia
        2. Recupera le prime max_pagine pagine trovate
        3. Costruisce per ognuna un dizionario compatibile con CorpusFever
        4. I metadati (numero_link, timestamp) sono simulati

        Il testo viene formattato con il campo 'lines' nel formato
        FEVER (frasi numerate) per compatibilità con estrai_frase().

        Args:
            query: testo della query (claim o domanda utente)

        Returns:
            Dizionario {id_doc: {lines, text, numero_link_entrata, timestamp}}
            Dizionario vuoto se Wikipedia non trova risultati.
        """
        corpus_runtime = {}

        # Cerca le pagine Wikipedia più rilevanti per la query
        titoli_trovati = self._cerca_titoli(query)

        for titolo in titoli_trovati[:self._max_pagine]:
            pagina = self._wiki.page(titolo)
            if not pagina.exists():
                continue

            testo   = pagina.text
            lines   = self._testo_in_formato_lines(testo)
            id_doc  = titolo.replace(' ', '_')

            corpus_runtime[id_doc] = {
                'lines':               lines,
                'text':                testo,
                # Proxy: pagine più lunghe = tendenzialmente più autorevoli
                'numero_link_entrata': len(testo),
                'timestamp':           TIMESTAMP_OGGI
            }

        return corpus_runtime

    def ottieni_documento(self, id_doc: str) -> dict:
        """
        Recupera una singola pagina Wikipedia dato il titolo.

        Args:
            id_doc: titolo della pagina (con underscore come spazi)

        Returns:
            Dizionario del documento, vuoto se non trovata
        """
        titolo = id_doc.replace('_', ' ')
        pagina = self._wiki.page(titolo)
        if not pagina.exists():
            return {}

        testo = pagina.text
        return {
            'lines':               self._testo_in_formato_lines(testo),
            'text':                testo,
            'numero_link_entrata': len(testo),
            'timestamp':           TIMESTAMP_OGGI
        }

    def numero_documenti(self) -> int:
        """
        In modalità web il corpus è dinamico — non ha dimensione fissa.

        Returns:
            0 — il corpus viene costruito a runtime per ogni query
        """
        return 0

    # ── Metodi privati ────────────────────────────────────────────

    def _cerca_titoli(self, query: str) -> list[str]:
        """
        Cerca i titoli delle pagine Wikipedia più rilevanti.

        Estrae le parole chiave dalla query (senza stopwords),
        le usa come termine di ricerca Wikipedia e restituisce
        i titoli delle pagine trovate.

        Usa le parole originali (non stemmiate) perché Wikipedia
        si aspetta termini leggibili, non stem.

        Args:
            query: testo della query

        Returns:
            Lista di titoli di pagine Wikipedia
        """
        # Usa le parole originali dalla query, non gli stem
        # Per la ricerca Wikipedia serve il testo leggibile
        import re
        from nltk.corpus import stopwords

        stop = set(stopwords.words('english'))
        parole = re.findall(r'\b[a-zA-Z]{3,}\b', query)
        parole_chiave = [p for p in parole if p.lower() not in stop]

        # Prova con la query completa prima, poi con parole chiave
        termini_ricerca = [query] + parole_chiave[:3]
        titoli = []

        for termine in termini_ricerca:
            pagina = self._wiki.page(termine)
            if pagina.exists() and pagina.title not in titoli:
                titoli.append(pagina.title)
            if len(titoli) >= self._max_pagine:
                break

        return titoli

    def _testo_in_formato_lines(self, testo: str) -> str:
        """
        Converte un testo libero nel formato 'lines' di FEVER.

        Il formato FEVER è: '0\\tFrase zero.\\n1\\tFrase uno.'
        Questo permette a ClassificatoreNLI di estrarre frasi
        specifiche anche dai documenti web, mantenendo la
        compatibilità con il resto della pipeline.

        La suddivisione in frasi è approssimativa (split per '.')
        ma sufficiente per il classificatore NLI che usa il
        testo come contesto, non per un'analisi sintattica precisa.

        Args:
            testo: testo grezzo della pagina Wikipedia

        Returns:
            Stringa nel formato FEVER lines
        """
        if not testo:
            return ''

        # Suddivide per punto e crea l'indice numerico
        frasi = [f.strip() for f in testo.split('.') if f.strip()]
        righe = [f'{i}\t{frase}.' for i, frase in enumerate(frasi[:200])]
        return '\n'.join(righe)