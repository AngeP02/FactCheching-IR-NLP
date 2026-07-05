import math
import re

from Sorgente.Indicizzazione.indice_invertito import IndiceInvertito
from Sorgente.Preprocessing.preprocesing import Preprocessore
from Sorgente.Recupero.metadati import GestoreMetadati
import config as cfg

PATTERN_LRB = re.compile(r'_-LRB-')
PATTERN_RRB = re.compile(r'-RRB-')
PATTERN_LSB = re.compile(r'_-LSB-')
PATTERN_RSB = re.compile(r'-RSB-')


def pulisci_titolo(titolo: str) -> str:
    titolo = PATTERN_LRB.sub(' (', titolo)
    titolo = PATTERN_RRB.sub(')', titolo)
    titolo = PATTERN_LSB.sub(' [', titolo)
    titolo = PATTERN_RSB.sub(']', titolo)
    titolo = titolo.replace('_', ' ').strip()
    return titolo


class CalcolatoreRanking:
    def __init__(
        self,
        indice: IndiceInvertito,
        preprocessore: Preprocessore,
        metadati: GestoreMetadati,
        k1: float = cfg.K1,
        b: float = cfg.B,
        alpha: float = cfg.PESO_AUTHORITY,
        beta: float = cfg.PESO_FRESHNESS
    ):
        self._indice = indice
        self._preprocessore = preprocessore
        self._metadati = metadati
        self._k1 = k1
        self._b = b
        self._alpha = alpha
        self._beta = beta

    def _calcola_idf(self, termine: str) -> float:
        N  = self._indice.numero_documenti
        df = self._indice.documenti_con_termine(termine)
        if df == 0:
            return 0.0
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def _calcola_tf_normalizzata(self,tf:int,lung_doc: int,avgdl:float) -> float:
        normalizzazione = 1 - self._b + self._b * (lung_doc / avgdl)
        return (tf * (self._k1 + 1)) / (tf + self._k1 * normalizzazione)

    def _calcola_bm25(self,token_query: list[str], avgdl:float) -> dict:
        punteggi = {}
        for termine in token_query:
            idf = self._calcola_idf(termine)
            posting = self._indice.cerca(termine)
            for id_doc, tf in posting.items():
                lung_doc = self._indice.lunghezza_documento(id_doc)
                tf_norm = self._calcola_tf_normalizzata(tf, lung_doc, avgdl)
                contributo = idf * tf_norm
                punteggi[id_doc] = punteggi.get(id_doc, 0.0) + contributo
        return punteggi

    def recupera_top_k(self, query: str, k: int = cfg.DOCUMENTI_DA_RECUPERARE) -> list[dict]:
        token_query = self._preprocessore.elabora(query)
        if not token_query:
            return []
        avgdl = self._indice.lunghezza_media_documenti()
        punteggi_bm25 = self._calcola_bm25(token_query, avgdl)
        if not punteggi_bm25:
            return []
        risultati = []
        for id_doc, score_bm25 in punteggi_bm25.items():
            auth = self._metadati.authority_score(id_doc)
            fresh = self._metadati.freshness_score(id_doc)
            score_finale = score_bm25 + self._alpha * auth + self._beta * fresh
            risultati.append({
                'id_doc': id_doc,
                'titolo': pulisci_titolo(id_doc),
                'punteggio_bm25': round(score_bm25, 4),
                'authority_score': round(auth, 4),
                'freshness_score': round(fresh, 4),
                'punteggio_finale': round(score_finale, 4)
            })
        risultati.sort(key=lambda x: -x['punteggio_finale'])
        return risultati[:k]

    def punteggio_singolo(self, query: str, id_doc: str) -> dict:
        token_query = self._preprocessore.elabora(query)
        avgdl = self._indice.lunghezza_media_documenti()
        score_bm25 = 0.0
        for termine in token_query:
            tf = self._indice.frequenza_documento(termine, id_doc)
            if tf > 0:
                idf = self._calcola_idf(termine)
                lung_doc = self._indice.lunghezza_documento(id_doc)
                tf_norm = self._calcola_tf_normalizzata(tf, lung_doc, avgdl)
                score_bm25 += idf * tf_norm
        auth  = self._metadati.authority_score(id_doc)
        fresh = self._metadati.freshness_score(id_doc)
        return {
            'id_doc': id_doc,
            'titolo': pulisci_titolo(id_doc),
            'punteggio_bm25': round(score_bm25, 4),
            'authority_score': round(auth, 4),
            'freshness_score': round(fresh, 4),
            'punteggio_finale': round(score_bm25 + self._alpha * auth + self._beta * fresh, 4)
        }