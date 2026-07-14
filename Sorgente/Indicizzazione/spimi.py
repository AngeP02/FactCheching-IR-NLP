import os
import pickle
from collections import defaultdict
from tqdm import tqdm
from Sorgente.Preprocessing.preprocesing import Preprocessore

class Spimi:

    def __init__(self, preprocessore: Preprocessore, dimensione_blocco: int = 500):
        self.preprocessore = preprocessore
        self.dimensione_blocco = dimensione_blocco

    def _costruisci_blocco(self, documenti_blocco: list[tuple[str, str]] ) -> tuple[dict, dict]:
        blocco = defaultdict(lambda: defaultdict(int))
        lunghezze = {}
        for id_doc, testo in documenti_blocco:
            token = self.preprocessore.elabora(testo)
            if not token:
                continue
            for termine in token:
                blocco[termine][id_doc] += 1
            lunghezze[id_doc] = len(token)
        blocco_ordinato = dict(sorted({t: dict(p) for t, p in blocco.items()}.items()))
        return blocco_ordinato, lunghezze

    def _salva_blocco(self, blocco: dict, lunghezze: dict, percorso: str) -> None:
        with open(percorso, 'wb') as f:
            pickle.dump({'blocco': blocco, 'lunghezze': lunghezze}, f)

    def _carica_blocco(self, percorso: str) -> tuple[dict, dict]:
        with open(percorso, 'rb') as f:
            dati = pickle.load(f)
        return dati['blocco'], dati['lunghezze']

    def _merge_blocchi(self, percorsi: list[str]) -> tuple[dict, dict]:
        indice_finale = defaultdict(lambda: defaultdict(int))
        lunghezze_totali = {}
        for percorso in tqdm(percorsi, desc='Merge blocchi'):
            blocco, lunghezze = self._carica_blocco(percorso)
            for termine, posting in blocco.items():
                for id_doc, freq in posting.items():
                    indice_finale[termine][id_doc] += freq
            lunghezze_totali.update(lunghezze)
        indice_ordinato = dict(sorted({t: dict(p) for t, p in indice_finale.items()}.items()))
        return indice_ordinato, lunghezze_totali

    def costruisci(self, documenti: list[tuple[str, str]], cartella_blocchi: str) -> tuple[dict, dict]:
        os.makedirs(cartella_blocchi, exist_ok=True)
        percorsi_blocchi = []
        numero_blocchi = (len(documenti) // self.dimensione_blocco) + 1
        print(f'SPIMI: {len(documenti):,} documenti ' f'in {numero_blocchi} blocchi da {self.dimensione_blocco}')
        for i in tqdm(range(0, len(documenti), self.dimensione_blocco), desc='Costruzione blocchi'):
            sotto_lista = documenti[i: i + self.dimensione_blocco]
            numero_blocco = i // self.dimensione_blocco
            percorso = os.path.join(cartella_blocchi, f'blocco_{numero_blocco:04d}.pkl')
            blocco, lunghezze = self._costruisci_blocco(sotto_lista)
            self._salva_blocco(blocco, lunghezze, percorso)
            percorsi_blocchi.append(percorso)
        print(f'Merge di {len(percorsi_blocchi)} blocchi')
        indice_finale, lunghezze_totali = self._merge_blocchi(percorsi_blocchi)
        print(f'Indice costruito: {len(indice_finale):,} termini unici, ' f'{len(lunghezze_totali):,} documenti')
        return indice_finale, lunghezze_totali