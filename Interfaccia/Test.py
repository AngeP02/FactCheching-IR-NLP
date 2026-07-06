import pickle
from Sorgente.Preprocessing.preprocesing import Preprocessore

with open('../Notebook/corpus_fever_arricchito.pkl', 'rb') as f:
    corpus = pickle.load(f)

doc = corpus.get('Barack_Obama', {})
lines = doc.get('lines', '')

# Cerca hawaii in tutte le frasi
print("FRASI CHE CONTENGONO 'hawaii':")
for riga in lines.split('\n'):
    parti = riga.split('\t', 1)
    if len(parti) == 2 and 'hawaii' in parti[1].lower():
        print(f"  Frase {parti[0]}: {parti[1][:150]}")