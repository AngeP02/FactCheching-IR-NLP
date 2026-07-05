# Parametri BM25
K1 = 1.5
B  = 0.75

# Pesi ranking esteso
PESO_AUTHORITY  = 0.3
PESO_FRESHNESS  = 0.2

# Recupero
DOCUMENTI_DA_RECUPERARE = 10  # top-K

# LLM
MODELLO_LLM    = "gpt-4o-mini"
TEMPERATURA    = 0.0
TOKEN_MASSIMI  = 50

# Dataset
SORGENTE_CORPUS = "fever"   # oppure "web"
LINGUA_CORPUS   = "en"

# Percorsi
PERCORSO_INDICE  = "dati/elaborati/indice.pkl"
PERCORSO_CORPUS  = "dati/grezzi/"

LABEL_SUPPORTA   = "SUPPORTS"
LABEL_SMENTISCE  = "REFUTES"
LABEL_INSUFFICIENTE = "NOT ENOUGH INFO"


# ── Label esatte (verificate dal dataset) ──────────────
LABEL_SUPPORTA      = "SUPPORTS"
LABEL_SMENTISCE     = "REFUTES"
LABEL_INSUFFICIENTE = "NOT ENOUGH INFO"
TUTTE_LE_LABEL      = [LABEL_SUPPORTA, LABEL_SMENTISCE, LABEL_INSUFFICIENTE]

# ── Dimensioni dataset (verificate) ───────────────────
CLAIM_TOTALI_TRAIN  = 145_449
CLAIM_TOTALI_DEV    = 19_000   # appross. dopo raggruppamento
PAGINE_WIKI_TRAIN   = 12_550   # corpus da indicizzare

# ── Split da usare ─────────────────────────────────────
SPLIT_INDICE        = "train"
SPLIT_SVILUPPO      = "campione_dev_bilanciato.csv"
SPLIT_VALUTAZIONE   = "dev_per_claim.csv"
SPLIT_TEST_FINALE   = "paper_test"   # usare UNA VOLTA SOLA

# ── BM25 ───────────────────────────────────────────────
K1 = 1.5
B  = 0.75

# ── Ranking esteso ─────────────────────────────────────
PESO_AUTHORITY  = 0.3
PESO_FRESHNESS  = 0.2

# ── Recupero ───────────────────────────────────────────
DOCUMENTI_DA_RECUPERARE = 10

# ── LLM ────────────────────────────────────────────────
MODELLO_LLM   = "gpt-4o-mini"
TEMPERATURA   = 0.0
TOKEN_MASSIMI = 50

# ── Corpus ─────────────────────────────────────────────
SORGENTE_CORPUS     = "fever"   # oppure "web"
PERCORSO_WIKI_PAGES = "dati/grezzi/wiki_pages/"
PERCORSO_INDICE     = "dati/elaborati/indice.pkl"

ESPANSIONE_ATTIVA = False