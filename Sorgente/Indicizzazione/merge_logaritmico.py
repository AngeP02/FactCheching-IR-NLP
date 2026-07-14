from collections import defaultdict
from Sorgente.Preprocessing.preprocesing import Preprocessore

class MergeLogaritmico:

    def __init__(self, preprocessore: Preprocessore):
        self.preprocessore = preprocessore
        self.livelli = []
        self.lunghezze_livelli = []
        self.dimensioni = []
        self.documenti_totali = 0

    def _indicizza_documento(self, id_doc: str, testo: str) -> tuple[dict, dict]:
        indice_doc = defaultdict(lambda: defaultdict(int))
        token = self.preprocessore.elabora(testo)
        for termine in token:
            indice_doc[termine][id_doc] += 1
        return dict(indice_doc), {id_doc: len(token)}

    def _fondi(self, indice_a: dict, lunghezze_a: dict, indice_b: dict, lunghezze_b: dict) -> tuple[dict, dict]:
        risultato   = defaultdict(lambda: defaultdict(int))
        lunghezze_r = {}
        for termine, posting in indice_a.items():
            for id_doc, freq in posting.items():
                risultato[termine][id_doc] += freq
        for termine, posting in indice_b.items():
            for id_doc, freq in posting.items():
                risultato[termine][id_doc] += freq
        lunghezze_r.update(lunghezze_a)
        lunghezze_r.update(lunghezze_b)
        return dict(risultato), lunghezze_r