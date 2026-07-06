from abc import ABC, abstractmethod


class CorpusBase(ABC):

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