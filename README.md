# Sistema di Fact-Checking automatico

Pipeline end-to-end per la verifica automatica di affermazioni in lingua inglese, basata su Information Retrieval e Natural Language Inference. 
Progetto per il corso di **Information Retrieval e NLP**

## Descrizione

Il sistema prende in input un'affermazione testuale e restituisce un verdetto tra tre classi:
1. **SUPPORTS** — l'evidenza disponibile supporta la claim;
2. **REFUTES** — l'evidenza contraddice la claim;
3. **NOT ENOUGH INFO** — le evidenze non sono sufficienti a determinare la veridicità.

Il verdetto è accompagnato da una misura di confidenza e da una spiegazione in linguaggio naturale. Il sistema è sviluppato e valutato sul dataset **FEVER** (Fact Extraction and VERification), benchmark di riferimento per il fact-checking su testi enciclopedici, costruito su ~5.4 milioni di pagine Wikipedia.

## Architettura

La pipeline di esecuzione è composta da:

```
Claim → Preprocessing → Espansione query (WordNet) → BM25 esteso → Re-ranking → Classificazione NLI → Aggregazione → Verdetto + Confidenza + Spiegazione
```
È composto da due moduli indipendenti eseguiti in sequenza:

1. **Modulo IR — Recupero delle evidenze** con le fasi di:
   1.1 **Preprocessing** — pulizia annotazioni FEVER, tokenizzazione NLTK, lowercase, rimozione punteggiatura e stopwords, stemming (Porter);
   1.2 **Espansione della query** — aggiunta di sinonimi WordNet (primo synset, opzionale);
   1.3 **BM25 esteso** — ranking con IDF smoothing Robertson, saturazione TF, penalizzazione lunghezza, authority score e freshness score;
   1.4 **Re-ranking** (opzionale) — cross-encoder transformer su coppie (claim, documento).
2. **Modulo NLI — Classificazione** con le fasi di:
   2.1 **Classificazione NLI** — GPT-4o-mini zero-shot con in-context learning;
   2.2 **Aggregazione** — combina score massimo del documento più rilevante per label e proporzione di consenso;
   2.3 **Spiegazione** — seconda chiamata GPT-4o-mini.

## Interfaccia

L'applicazione Streamlit è strutturata in quattro pagine:
1. **Home** — inserimento claim, verdetto, spiegazione e feed delle ultime 12 verifiche;
2. **Analisi tecnica** — token preprocessati, distribuzione pesi per label, documenti recuperati con score e estratto evidenza;
3. **Benchmark** — valutazione automatica su campione FEVER configurabile;
4. **Impostazioni** — K documenti, sorgente corpus (FEVER locale o Wikipedia live), re-ranking, espansione WordNet.
