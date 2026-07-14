import pickle
from Sorgente.Corpus.corpus_base import CorpusBase

class CorpusFever(CorpusBase):

    def __init__(self, percorso_corpus: str):
        with open(percorso_corpus, 'rb') as f:
            self._corpus: dict = pickle.load(f)
        self._corpus = {k: v for k, v in self._corpus.items() if k.strip()}

    def numero_documenti(self) -> int:
        return len(self._corpus)