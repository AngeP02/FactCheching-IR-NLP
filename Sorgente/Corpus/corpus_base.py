"""
corpus_base.py
════════════════════════════════════════════════════════════════════
Interfaccia comune per le sorgenti di documenti.

Responsabilità:
    Definire il contratto che ogni sorgente di corpus deve rispettare,
    garantendo che la pipeline possa usare indifferentemente il corpus
    FEVER o il corpus web senza modifiche al codice.

Perché un'interfaccia comune:
    La pipeline ha bisogno di documenti da dare al classificatore NLI.
    Da dove arrivano questi documenti (dataset fisso o web live) è un
    dettaglio implementativo che non deve influenzare il resto del sistema.

    Questo è il principio di inversione delle dipendenze (SOLID):
    la pipeline dipende dall'astrazione CorpusBase, non dalle
    implementazioni concrete CorpusFever o CorpusWeb.

    Conseguenza pratica: cambiare sorgente significa cambiare una
    riga in configurazione.py, nessuna modifica alla pipeline.

Struttura del documento restituito:
    Ogni implementazione deve restituire dizionari con questa struttura:
    {
        'id_doc': {
            'lines':               str,   # frasi numerate (formato FEVER)
            'text':                str,   # testo completo
            'numero_link_entrata': int,   # per authority_score
            'timestamp':           str    # per freshness_score (ISO format)
        }
    }

Utilizzo:
    Non si istanzia direttamente — usare CorpusFever o CorpusWeb.
    La pipeline usa CorpusBase come tipo nelle annotazioni.
════════════════════════════════════════════════════════════════════
"""

from abc import ABC, abstractmethod


class CorpusBase(ABC):
    """
    Interfaccia astratta per le sorgenti di documenti.

    Definisce il contratto che CorpusFever e CorpusWeb devono
    rispettare per essere intercambiabili nella pipeline.
    """

    @abstractmethod
    def ottieni_documenti(self, query: str) -> dict:
        """
        Restituisce i documenti rilevanti per una query.

        In CorpusFever: restituisce i documenti pre-caricati.
        In CorpusWeb:   recupera documenti Wikipedia in tempo reale.

        Args:
            query: testo della query (claim o domanda utente)

        Returns:
            Dizionario {id_doc: {lines, text, numero_link_entrata, timestamp}}
            Dizionario vuoto se nessun documento trovato.
        """

    @abstractmethod
    def ottieni_documento(self, id_doc: str) -> dict:
        """
        Restituisce un singolo documento dato il suo identificatore.

        Args:
            id_doc: identificatore del documento

        Returns:
            Dizionario {lines, text, numero_link_entrata, timestamp}
            Dizionario vuoto se il documento non esiste.
        """

    @abstractmethod
    def numero_documenti(self) -> int:
        """
        Restituisce il numero di documenti disponibili nella sorgente.

        Returns:
            Numero intero di documenti.
        """