from abc import ABC, abstractmethod


class CorpusBase(ABC):
    @abstractmethod
    def numero_documenti(self) -> int:
        """
        Restituisce il numero di documenti disponibili nella sorgente.

        Returns:
            Numero intero di documenti.
        """