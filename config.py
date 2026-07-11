"""
configurazione.py
════════════════════════════════════════════════════════════════════
Unica fonte di verità per tutti i parametri del progetto.

Responsabilità:
    Centralizzare tutti i parametri del sistema in un unico file
    così da non avere magic number sparsi nel codice.
    Qualsiasi modifica ai parametri avviene qui e si propaga
    automaticamente a tutti i moduli che importano questo file.

Come usarlo:
    import configurazione as cfg
    k1 = cfg.K1
    modello = cfg.MODELLO_LLM
════════════════════════════════════════════════════════════════════
"""

# ── Sorgente corpus ───────────────────────────────────────────────
# 'fever' → usa il corpus FEVER pre-costruito e l'indice su disco
# 'web'   → recupera documenti Wikipedia in tempo reale


import os

# Ottieni la cartella dove si trova app.py (la cartella 'Interfaccia')
cartella_corrente = os.path.dirname(os.path.abspath(__file__))

# Sali di un livello (esce da 'Interfaccia' e va in 'PROGETTO')
cartella_progetto = os.path.dirname(cartella_corrente)

SORGENTE_CORPUS = 'fever'

# ── Percorsi file ─────────────────────────────────────────────────
PERCORSO_CORPUS_ARRICCHITO = os.path.join(cartella_progetto, 'PROGETTO/Notebook','corpus_fever_arricchito.pkl')
PERCORSO_INDICE = os.path.join(cartella_progetto, 'PROGETTO/Dati/Elaborati','indice_fever.pkl')
PERCORSO_BLOCCHI_SPIMI= os.path.join(cartella_progetto, 'PROGETTO/Dati/Elaborati/blocchi_spimi/')
PERCORSO_CAMPIONE_DEV = os.path.join(cartella_progetto, 'PROGETTO/Notebook','campione_dev_bilanciato.csv')
PERCORSO_DEV_COMPLETO = os.path.join(cartella_progetto, 'PROGETTO/Notebook', 'dev_per_claim.csv')

# ── Parametri BM25 ────────────────────────────────────────────────
# k1: controlla la saturazione della term frequency
#     k1 alto → la tf cresce più lentamente verso la saturazione
#     valore standard per testo enciclopedico: 1.2 - 2.0
K1 = 1.5

# b: controlla la normalizzazione per lunghezza documento
#    b=0 → nessuna normalizzazione
#    b=1 → normalizzazione completa
#    valore standard: 0.75
B  = 0.75

# ── Pesi ranking esteso ───────────────────────────────────────────
# Peso del authority_score nel ranking BM25 esteso
# punteggio_finale = BM25 + PESO_AUTHORITY * auth + PESO_FRESHNESS * fresh
PESO_AUTHORITY = 0.3
PESO_FRESHNESS = 0.2

# ── Recupero documenti ────────────────────────────────────────────
# Numero di documenti top-K recuperati da BM25 per ogni query
# Più alto = più recall, più costo API per la classificazione NLI
DOCUMENTI_DA_RECUPERARE = 5

# ── Espansione query ──────────────────────────────────────────────
# False → disabilitata per default (impatto marginale su FEVER)
# True  → aggiunge sinonimi WordNet ai termini della query
# Validato nel notebook 04: Recall@K quasi invariato con/senza
ESPANSIONE_ATTIVA = False

# Numero massimo di sinonimi WordNet aggiunti per termine
MAX_SINONIMI_ESPANSIONE = 3

# ── Modello LLM ───────────────────────────────────────────────────
# Modello OpenAI usato per la classificazione NLI
MODELLO_LLM = 'gpt-4o-mini'

# ── Re-ranking neurale ────────────────────────────────────────────
# True → usa cross-encoder dopo BM25 per riordinare i documenti
# False → usa solo BM25 (più veloce, comportamento originale)
RERANKING_ATTIVO = True

# Quanti candidati BM25 passare al reranker
# Deve essere > DOCUMENTI_DA_RECUPERARE per dare margine al reranker
CANDIDATI_RERANKING = 20

# Temperatura 0 → output deterministico e riproducibile
# Fondamentale per la valutazione scientifica
TEMPERATURA = 0.0

# Numero massimo di token nella risposta dell'LLM
# La label è al massimo 3 parole → 10 token sono sufficienti
TOKEN_MASSIMI = 10

# Numero massimo di caratteri del testo evidenza inviato all'LLM
# Limita i costi API mantenendo il contesto sufficiente
MAX_CARATTERI_EVIDENZA = 3000

# ── Label FEVER ───────────────────────────────────────────────────
# Nomi esatti delle label nel dataset FEVER v1.0
# Verificati nel notebook 00_studio_dataset
LABEL_SUPPORTA      = 'SUPPORTS'
LABEL_SMENTISCE     = 'REFUTES'
LABEL_INSUFFICIENTE = 'NOT ENOUGH INFO'
TUTTE_LE_LABEL      = [LABEL_SUPPORTA, LABEL_SMENTISCE, LABEL_INSUFFICIENTE]

# ── Dimensioni dataset (verificate nel notebook 00) ───────────────
CLAIM_TOTALI_TRAIN  = 145_449
CLAIM_TOTALI_DEV    = 19_033
PAGINE_WIKI_CORPUS  = 14_364

# ── Split da usare ────────────────────────────────────────────────
# Questi split vanno usati con disciplina:
# - SPLIT_SVILUPPO:   durante lo sviluppo e il tuning
# - SPLIT_VALUTAZIONE: per i risultati intermedi
# - SPLIT_TEST_FINALE: UNA VOLTA SOLA per i risultati finali
SPLIT_INDICE        = 'train'
SPLIT_SVILUPPO      = 'campione_dev_bilanciato.csv'   # 1500 claim bilanciate
SPLIT_VALUTAZIONE   = 'dev_per_claim.csv'             # ~19K claim
SPLIT_TEST_FINALE   = 'paper_test'                    # NON usare durante lo sviluppo

# ── Valutazione ───────────────────────────────────────────────────
# Valori di K per le metriche P@K, R@K, F1@K
K_VALORI_VALUTAZIONE = [1, 3, 5, 10]

# Numero di claim da usare per la valutazione rapida durante lo sviluppo
CLAIM_VALUTAZIONE_RAPIDA = 100

# ── Corpus web ────────────────────────────────────────────────────
# Numero massimo di pagine Wikipedia recuperate per ogni query web
MAX_PAGINE_WEB = 5

# ── SPIMI ─────────────────────────────────────────────────────────
# Numero di documenti per blocco durante la costruzione dell'indice
# Bilanciamento tra uso RAM e numero di file su disco
DIMENSIONE_BLOCCO_SPIMI = 500