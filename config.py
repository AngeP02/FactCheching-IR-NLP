import os

cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.dirname(cartella_corrente)

SORGENTE_CORPUS = 'fever'
PERCORSO_CORPUS_ARRICCHITO = os.path.join(cartella_progetto, 'PROGETTO/Notebook','corpus_fever_arricchito.pkl')
PERCORSO_INDICE = os.path.join(cartella_progetto, 'PROGETTO/Dati/Elaborati','indice_fever.pkl')

K1 = 1.5
B = 0.75
PESO_AUTHORITY = 0.3
PESO_FRESHNESS = 0.2
DOCUMENTI_DA_RECUPERARE = 5
ESPANSIONE_ATTIVA = False
MODELLO_LLM = 'gpt-4o-mini'
NOME_MODELLO_CROSS_ENCODER = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
RERANKING_ATTIVO = True
CANDIDATI_RERANKING = 20
TEMPERATURA = 0.0
TOKEN_MASSIMI = 10

LABEL_SUPPORTA = 'SUPPORTS'
LABEL_SMENTISCE = 'REFUTES'
LABEL_INSUFFICIENTE = 'NOT ENOUGH INFO'
TUTTE_LE_LABEL = [LABEL_SUPPORTA, LABEL_SMENTISCE, LABEL_INSUFFICIENTE]

K_VALORI_VALUTAZIONE = [1, 3, 5, 10]