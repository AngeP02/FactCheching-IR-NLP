from sentence_transformers import CrossEncoder
import config as cfg


class Reranker:
    NOME_MODELLO_CROSS_ENCODER = cfg.NOME_MODELLO_CROSS_ENCODER
    def __init__(self):
        print(f'  Caricamento cross-encoder: {self.NOME_MODELLO_CROSS_ENCODER}')
        self._modello = CrossEncoder(self.NOME_MODELLO_CROSS_ENCODER)
        self._max_caratteri = 512
        print('  ✓ Cross-encoder caricato')

    def _estrai_testo_documento(self, id_doc: str, corpus: dict) -> str:
        doc = corpus.get(id_doc, {})
        testo = doc.get('text', '') or ''
        return testo[:self._max_caratteri]

    def reordina(self, claim: str, documenti: list[dict], corpus: dict, top_k: int = None) -> list[dict]:
        if not documenti:
            return []
        coppie = []
        for doc in documenti:
            testo = self._estrai_testo_documento(doc['id_doc'], corpus)
            coppie.append((claim, testo))
        scores = self._modello.predict(coppie)
        documenti_con_score = []
        for doc, score in zip(documenti, scores):
            doc_arricchito = dict(doc)
            doc_arricchito['score_reranking'] = round(float(score), 4)
            documenti_con_score.append(doc_arricchito)
        documenti_riordinati = sorted(
            documenti_con_score,
            key=lambda x: -x['score_reranking']
        )
        if top_k:
            return documenti_riordinati[:top_k]
        return documenti_riordinati
