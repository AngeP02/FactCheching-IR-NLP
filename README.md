# Sistema di Fact-Checking automatico

Pipeline end-to-end per la verifica automatica di affermazioni in lingua inglese, basata su Information Retrieval e Natural Language Inference.  
Progetto per il corso di **Information Retrieval e NLP**

---

## Descrizione

Il sistema prende in input un'affermazione testuale e restituisce un verdetto tra tre classi:

- **SUPPORTS** — l'evidenza disponibile supporta la claim
- **REFUTES** — l'evidenza contraddice la claim
- **NOT ENOUGH INFO** — le evidenze non sono sufficienti a determinare la veridicità

Il verdetto è accompagnato da una misura di confidenza e da una spiegazione in linguaggio naturale.

Il sistema è sviluppato e valutato sul dataset **FEVER** (Fact Extraction and VERification), benchmark di riferimento per il fact-checking su testi enciclopedici, costruito su ~5.4 milioni di pagine Wikipedia.

---

## Architettura

La pipeline è composta da due moduli indipendenti in sequenza:

```
Claim → Preprocessing → Espansione query (WordNet) → BM25 esteso → Re-ranking → Classificazione NLI → Aggregazione → Verdetto + Confidenza + Spiegazione
```

**Modulo IR — Recupero delle evidenze**

1. **Preprocessing** — pulizia annotazioni FEVER, tokenizzazione NLTK, lowercase, rimozione punteggiatura e stopwords, stemming (Porter)
2. **Espansione della query** — aggiunta di sinonimi WordNet (primo synset, opzionale)
3. **BM25 esteso** — ranking con IDF smoothing Robertson, saturazione TF, penalizzazione lunghezza, authority score (backlink, log-normalizzati) e freshness score (timestamp stimato); pesi α=0.3, β=0.2
4. **Re-ranking** — cross-encoder transformer su coppie (claim, documento), opzionale

**Modulo NLI — Classificazione**

5. **Classificazione NLI** — GPT-4o-mini zero-shot con in-context learning; k=5 chiamate API separate, una per documento
6. **Aggregazione** — combina score massimo del documento più rilevante per label (peso 0.6) e proporzione di consenso (peso 0.4); penalizzazione 0.5× su NOT ENOUGH INFO
7. **Spiegazione** — seconda chiamata GPT-4o-mini (temperatura 0.3, max 100 token)

---

## Interfaccia

L'applicazione Streamlit è strutturata in quattro pagine:

- **Home** — inserimento claim, verdetto, spiegazione e feed delle ultime 12 verifiche
- **Analisi tecnica** — token preprocessati, distribuzione pesi per label, documenti recuperati con score e estratto evidenza
- **Benchmark** — valutazione automatica su campione FEVER configurabile (slider per n° claim, scelta dello split)
- **Impostazioni** — K documenti, sorgente corpus (FEVER locale o Wikipedia live), re-ranking, espansione WordNet
