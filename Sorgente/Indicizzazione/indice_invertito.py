import pickle
from collections import defaultdict

class IndiceInvertito:

    def __init__(self):
        self._indice = defaultdict(lambda: defaultdict(int))
        self._lunghezze_documenti = {}
        self._numero_documenti = 0

    def aggiungi_documento(self, id_doc: str, token: list[str]) -> None:
        if not token:
            return
        for termine in token:
            self._indice[termine][id_doc] += 1
        self._lunghezze_documenti[id_doc] = len(token)
        self._numero_documenti += 1

    def cerca(self, termine: str) -> dict:
        return dict(self._indice.get(termine, {}))

    def frequenza_documento(self, termine: str, id_doc: str) -> int:
        return self._indice.get(termine, {}).get(id_doc, 0)

    def documenti_con_termine(self, termine: str) -> int:
        return len(self._indice.get(termine, {}))

    def lunghezza_documento(self, id_doc: str) -> int:
        return self._lunghezze_documenti.get(id_doc, 0)

    def lunghezza_media_documenti(self) -> float:
        if not self._lunghezze_documenti:
            return 0.0
        return sum(self._lunghezze_documenti.values()) / len(self._lunghezze_documenti)

    @property
    def numero_documenti(self) -> int:
        return self._numero_documenti

    @property
    def numero_termini(self) -> int:
        return len(self._indice)

    def carica(self, percorso: str) -> None:
        with open(percorso, 'rb') as f:
            stato = pickle.load(f)
        self._indice = defaultdict(lambda: defaultdict(int))
        for termine, posting in stato['indice'].items():
            for id_doc, freq in posting.items():
                self._indice[termine][id_doc] = freq
        self._lunghezze_documenti = stato['lunghezze_documenti']
        self._numero_documenti    = stato['numero_documenti']