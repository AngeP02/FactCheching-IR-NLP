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

    def inserisci(self, id_doc: str, testo: str) -> None:
        nuovo_indice, nuove_lunghezze = self._indicizza_documento(id_doc, testo)
        nuova_dim = 1
        i = 0
        while i < len(self.livelli) and self.livelli[i] is not None:
            nuovo_indice, nuove_lunghezze = self._fondi(self.livelli[i], self.lunghezze_livelli[i], nuovo_indice, nuove_lunghezze)
            nuova_dim += self.dimensioni[i]

            self.livelli[i] = None
            self.lunghezze_livelli[i] = {}
            self.dimensioni[i] = 0
            i += 1

        if i == len(self.livelli):
            # Crea un nuovo livello
            self.livelli.append(nuovo_indice)
            self.lunghezze_livelli.append(nuove_lunghezze)
            self.dimensioni.append(nuova_dim)
        else:
            # Occupa il livello libero esistente
            self.livelli[i] = nuovo_indice
            self.lunghezze_livelli[i] = nuove_lunghezze
            self.dimensioni[i] = nuova_dim

        self.documenti_totali += 1

    def ottieni_indice_completo(self) -> tuple[dict, dict]:
        indice_acc = {}
        lunghezze_acc = {}
        for livello, lunghezze in zip(self.livelli, self.lunghezze_livelli):
            if livello is not None:
                indice_acc, lunghezze_acc = self._fondi(indice_acc, lunghezze_acc, livello, lunghezze)
        return indice_acc, lunghezze_acc

    def stato(self) -> None:
        print(f'Documenti totali inseriti: {self.documenti_totali}')
        print(f'Rappresentazione binaria:  ' f'{self.documenti_totali:0{max(1,len(self.livelli))}b}')
        print('Livelli:')
        for i, (livello, dim) in enumerate(zip(self.livelli, self.dimensioni)):
            occupato = livello is not None
            stato = f'{dim} doc (max {2**i})' if occupato else 'vuoto'
            bit = '1' if occupato else '0'
            print(f'  Z{i}  [bit={bit}]  {stato}')