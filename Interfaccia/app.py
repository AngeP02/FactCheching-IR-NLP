import os
import sys

# Configurazione path
cartella_interfaccia = os.path.dirname(os.path.abspath(__file__))
cartella_progetto = os.path.dirname(cartella_interfaccia)
if cartella_progetto not in sys.path:
    sys.path.insert(0, cartella_progetto)

import streamlit as st
from dotenv import load_dotenv
import config as cfg
from Sorgente.Pipeline import Pipeline

load_dotenv()

st.set_page_config(
    page_title='Fact Checker AI',
    page_icon='🤖',
    layout='wide'
)


# --- INIZIALIZZAZIONE E CACHE ---

@st.cache_resource
def carica_pipeline(api_key: str) -> Pipeline:
    return Pipeline(api_key=api_key)


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_stats" not in st.session_state:
        st.session_state.last_stats = None


def get_ui_elements(verdetto: str) -> tuple[str, str]:
    mappa = {
        'SUPPORTS': ('#2ecc71', '✅'),
        'REFUTES': ('#e74c3c', '❌'),
        'NOT ENOUGH INFO': ('#f39c12', '⚠️')
    }
    return mappa.get(verdetto, ('#95a5a6', '❓'))


# --- RENDER DELLE PAGINE ---

def pagina_chatbot(pipeline: Pipeline):
    st.title('🤖 Fact Checker AI')
    st.caption("Chiedimi di verificare un'affermazione. Cercherò le prove e ti darò un verdetto.")

    # Mostra la cronologia della chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

            # CORREZIONE 2: Ora disegna l'expander solo se ci sono effettivamente fonti salvate
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📚 Mostra le fonti analizzate"):
                    for i, doc in enumerate(msg["sources"], 1):
                        st.markdown(f"**{i}. {doc['titolo']}**")
                        st.info(doc['evidenza'])

    # Input utente
    if prompt := st.chat_input("Es: L'attuale presidente della Repubblica Italiana è Sergio Mattarella"):
        # Mostra messaggio utente
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Risposta assistente
        with st.chat_message("assistant"):
            with st.spinner("Sto cercando le evidenze e ragionando..."):
                try:
                    stats = None
                    if cfg.ESPANSIONE_ATTIVA and cfg.SORGENTE_CORPUS == 'fever':
                        try:
                            stats = pipeline._espansore.statistiche_espansione(prompt)
                        except Exception:
                            pass

                    # Esecuzione pipeline
                    risultato = pipeline.verifica(prompt)

                    # Salva in sessione per la pagina sviluppatori
                    st.session_state.last_result = risultato
                    st.session_state.last_stats = stats

                    verdetto = risultato['verdetto']
                    confidenza = risultato['confidenza']
                    colore, icona = get_ui_elements(verdetto)

                    risposta_html = (
                        f"Il mio verdetto è: <strong style='color:{colore}; font-size:1.2em'>"
                        f"{icona} {verdetto}</strong> (Confidenza: {confidenza:.1f}%)"
                    )

                    st.markdown(risposta_html, unsafe_allow_html=True)

                    # Spiegazione generata dal LLM (componente RAG generativa)
                    spiegazione = risultato.get('spiegazione', '')
                    if spiegazione:
                        st.markdown(f"_{spiegazione}_")

                    # CORREZIONE 1: La chiave giusta restituita dalla tua pipeline è 'documenti', non 'dettagli'
                    documenti = risultato.get('documenti', [])

                    if documenti:
                        with st.expander("📚 Mostra le fonti analizzate"):
                            for i, doc in enumerate(documenti, 1):
                                st.markdown(f"**{i}. {doc['titolo']}**")
                                st.info(doc['evidenza'])
                    else:
                        st.warning("Non ho trovato documenti pertinenti per questa ricerca.")

                    # Salva messaggio in cronologia
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": risposta_html + (f"\n\n_{spiegazione}_" if spiegazione else ""),
                        "sources": documenti
                    })

                except Exception as e:
                    st.error(f'Ops! Si è verificato un errore: {e}')


def pagina_sviluppatori():
    st.title('⚙️ Dietro le Quinte (Developer Mode)')
    st.markdown(
        "In questa pagina puoi ispezionare i punteggi **Information Retrieval (IR)** e le logiche **Natural Language Processing (NLP)** dell'ultima query elaborata.")
    st.divider()

    risultato = st.session_state.last_result

    if not risultato:
        st.info("Esegui prima una ricerca nella pagina Chatbot per vedere i dettagli qui.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔍 NLP: Dettagli Classificazione")
        pesi = risultato.get('pesi_label', {})
        if pesi:
            st.write("**Pesi aggregati pesati dal BM25:**")
            peso_tot = sum(pesi.values()) or 1
            st.progress(pesi.get('SUPPORTS', 0) / peso_tot, text=f"SUPPORTS ({pesi.get('SUPPORTS', 0):.2f})")
            st.progress(pesi.get('REFUTES', 0) / peso_tot, text=f"REFUTES ({pesi.get('REFUTES', 0):.2f})")
            st.progress(pesi.get('NOT ENOUGH INFO', 0) / peso_tot,
                        text=f"NOT ENOUGH INFO ({pesi.get('NOT ENOUGH INFO', 0):.2f})")

        # Spiegazione generativa — componente RAG
        spiegazione = risultato.get('spiegazione', '')
        if spiegazione:
            st.write("**📝 Spiegazione generata dal LLM:**")
            st.info(spiegazione)

        stats = st.session_state.last_stats
        if stats and stats['n_aggiunti'] > 0:
            st.write("**Espansione Query (WordNet):**")
            st.code(f"Originali: {stats['token_originali']}\nEspansi: {stats['sinonimi_aggiunti']}")

    with col2:
        st.subheader("📊 IR: Ranking e Punteggi BM25")
        # CORREZIONE: Aggiornato anche qui a 'documenti'
        documenti = risultato.get('documenti', [])
        if documenti:
            for doc in documenti:
                colore, icona = get_ui_elements(doc['verdetto'])
                st.markdown(f"**{doc['titolo']}**")

                # Mostra entrambi gli score se il reranking è attivo
                if 'score_reranking' in doc:
                    st.caption(
                        f"BM25: `{doc['punteggio']:.4f}` | "
                        f"Re-ranking: `{doc['score_reranking']:.4f}` | "
                        f"Verdetto: {icona} {doc['verdetto']}"
                    )
                else:
                    st.caption(
                        f"Score BM25: `{doc['punteggio']:.4f}` | "
                        f"Verdetto: {icona} {doc['verdetto']}"
                    )
                st.divider()
        else:
            st.write("Nessun documento recuperato.")


# --- MAIN APP ---

def main():
    init_session_state()
    api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        st.error("⚠️ Nessuna API Key trovata. Imposta OPENAI_API_KEY nel file .env")
        return

    try:
        pipeline = carica_pipeline(api_key)
    except FileNotFoundError as e:
        st.error(f'File non trovato: {e}\n\nEsegui prima i notebook di setup (00 e 02).')
        return

    with st.sidebar:
        st.header('🧭 Navigazione')
        pagina_corrente = st.radio("Seleziona vista:", ["💬 Chatbot", "⚙️ Sviluppatori"], label_visibility="collapsed")

        st.divider()
        st.header('⚙️ Configurazione')
        sorgente = st.radio('Sorgente Dati', ['fever', 'web'],
                            format_func=lambda x: '📚 FEVER (Locale)' if x == 'fever' else '🌐 Wikipedia Live')
        cfg.SORGENTE_CORPUS = sorgente

        cfg.DOCUMENTI_DA_RECUPERARE = st.slider('Top-K Documenti (IR)', 1, 10, cfg.DOCUMENTI_DA_RECUPERARE)
        cfg.ESPANSIONE_ATTIVA = st.toggle('Attiva WordNet', value=cfg.ESPANSIONE_ATTIVA)

        st.divider()
        st.caption(f"LLM: {cfg.MODELLO_LLM}")

        # Premere questo bottone svuota la cronologia e riporta a zero l'app
        if st.button("🗑️ Svuota Chat"):
            st.session_state.messages = []
            st.session_state.last_result = None
            st.session_state.last_stats = None
            st.rerun()

    if pagina_corrente == "💬 Chatbot":
        pagina_chatbot(pipeline)
    else:
        pagina_sviluppatori()
        # ──────────────────────────────────────────────────────────────────────
        # VALUTAZIONE GLOBALE DEL SISTEMA (BENCHMARK DATASET)
        # ──────────────────────────────────────────────────────────────────────
        st.divider()
        st.subheader("📊 Valutazione Globale del Sistema")
        st.write(
            "Esegui un test controllato su un campione del dataset FEVER per calcolare parametri come MAP, Recall e Accuratezza NLI.")

        # Input interattivi per lo sviluppatore
        campione_size = st.number_input("Numero di claim da testare nel benchmark", min_value=2, max_value=200,
                                        value=10)
        path_fever_test = st.text_input("Percorso del file Ground Truth (JSONL)", value="fever_dev_sample.jsonl")

        if st.button("🚀 Avvia Benchmark su Dataset"):
            if not os.path.exists(path_fever_test):
                st.error(f"❌ File non trovato al percorso specificato: '{path_fever_test}'. Verifica la cartella.")
            else:
                import json
                with st.spinner(
                        "Valutazione in corso... Il sistema sta elaborando i claim e confrontando i risultati con il Ground Truth."):

                    # 1. Caricamento del campione dal file JSONL di FEVER
                    dati_test = []
                    with open(path_fever_test, 'r', encoding='utf-8') as f:
                        for i, linea in enumerate(f):
                            if i >= campione_size:
                                break
                            item = json.loads(linea)

                            # Estraiamo gli ID dei documenti rilevanti reali indicati nel dataset FEVER
                            true_docs = []
                            if 'evidence' in item:
                                for annotation in item['evidence']:
                                    for ev in annotation:
                                        if len(ev) > 2 and ev[2] is not None:
                                            true_docs.append(ev[2])

                            dati_test.append({
                                'claim': item.get('claim', ''),
                                'true_label': item.get('label', 'NOT ENOUGH INFO'),
                                'true_doc_ids': list(set(true_docs))  # Rimuove duplicati di annotazione
                            })

                    lista_risultati_ir = []
                    label_vere = []
                    label_predette = []

                    # Barra di avanzamento visiva su Streamlit
                    progress_bar = st.progress(0)

                    # 2. Loop di predizione sui claim di test
                    for idx, item in enumerate(dati_test):
                        claim = item['claim']

                        # Eseguiamo la pipeline. Il metodo verifica aggiornerà st.session_state.last_result
                        # o restituirà il dizionario a seconda di come è mappato in Pipeline.py
                        pipeline.verifica(claim)
                        risultato = st.session_state.last_result

                        if risultato:
                            # Estraiamo gli ID dei documenti recuperati dal nostro sistema (BM25 + Reranker)
                            documenti_recuperati = [doc['id_doc'] for doc in risultato.get('dettagli', [])]

                            # Accumulo dati per metriche IR
                            lista_risultati_ir.append({
                                'recuperati': documenti_recuperati,
                                'rilevanti': item['true_doc_ids']
                            })

                            # Accumulo dati per metriche NLP
                            label_vere.append(item['true_label'])
                            label_predette.append(risultato.get('verdetto', 'NOT ENOUGH INFO'))

                        progress_bar.progress((idx + 1) / len(dati_test))

                    # 3. Calcolo effettivo mediante i tuoi moduli di valutazione
                    try:
                        from Sorgente.Valutazione.metriche_ir import MetricheIR
                        calcolatore_ir = MetricheIR()
                        report_ir = calcolatore_ir.valuta_sistema(lista_risultati_ir, k_valori=[1, 3, 5])

                        from Sorgente.Valutazione.metriche_nlp import MetricheNLP
                        calcolatore_nlp = MetricheNLP()
                        report_nlp = calcolatore_nlp.valuta_sistema(label_vere, label_predette)

                        # 4. Rendering delle metriche nell'interfaccia Streamlit
                        st.success("✅ Valutazione completata!")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("### 🔍 Performance del Modulo IR")
                            st.metric("Mean Average Precision (MAP)", f"{report_ir.get('MAP', 0.0):.4f}")
                            st.metric("Recall@3", f"{report_ir.get('R@3', 0.0):.4f}")
                            st.metric("Precision@3", f"{report_ir.get('P@3', 0.0):.4f}")

                        with col2:
                            st.markdown("### 🧠 Performance della Classificazione NLI")
                            st.metric("Accuracy Globale", f"{report_nlp.get('accuracy', 0.0):.4f}")
                            st.metric("F1 Macro", f"{report_nlp.get('f1_macro', 0.0):.4f}")
                            st.metric("Kappa di Cohen", f"{report_nlp.get('kappa_cohen', 0.0):.4f}")

                        # Visualizzazione della tabella dettagliata per singola classe
                        st.write("#### 📊 Breakdown per Classe (Dettaglio NLP)")
                        st.dataframe(report_nlp.get('per_classe'))

                    except Exception as e:
                        st.error(f"Si è verificato un errore durante il calcolo delle metriche: {e}")


if __name__ == '__main__':
    main()