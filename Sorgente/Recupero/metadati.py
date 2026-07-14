import numpy as np
from datetime import datetime

DATA_RIFERIMENTO = datetime(2004, 1, 1)

class GestoreMetadati:
    def __init__(self, corpus_arricchito: dict):
        self._authority_scores, self._freshness_scores = \
            self._calcola_tutti_i_punteggi(corpus_arricchito)

    def _timestamp_in_giorni(self, timestamp_str: str) -> float:
        if not timestamp_str:
            return 0.0
        try:
            dt = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%SZ')
            return max(0.0, (dt - DATA_RIFERIMENTO).days)
        except ValueError:
            return 0.0

    def _normalizza_min_max(self, valori: np.ndarray) -> np.ndarray:
        minimo = valori.min()
        massimo = valori.max()
        if massimo == minimo:
            return np.full_like(valori, 0.5, dtype=float)
        return (valori - minimo) / (massimo - minimo)

    def _calcola_tutti_i_punteggi(self, corpus_arricchito: dict) -> tuple[dict, dict]:
        titoli = list(corpus_arricchito.keys())
        link_grezzi = np.array(
            [corpus_arricchito[t]['numero_link_entrata'] for t in titoli],
            dtype=float
        )
        link_logaritmici = np.log1p(link_grezzi)
        giorni_grezzi = np.array(
            [self._timestamp_in_giorni(corpus_arricchito[t]['timestamp'])
             for t in titoli],
            dtype=float
        )
        authority_norm = self._normalizza_min_max(link_logaritmici)
        freshness_norm = self._normalizza_min_max(giorni_grezzi)
        authority_scores = dict(zip(titoli, authority_norm.tolist()))
        freshness_scores = dict(zip(titoli, freshness_norm.tolist()))
        return authority_scores, freshness_scores

    def authority_score(self, id_doc: str) -> float:
        return self._authority_scores.get(id_doc, 0.0)

    def freshness_score(self, id_doc: str) -> float:
        return self._freshness_scores.get(id_doc, 0.0)