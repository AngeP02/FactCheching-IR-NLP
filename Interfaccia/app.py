import os
import sys

cartella_interfaccia = os.path.dirname(os.path.abspath(__file__))
cartella_progetto    = os.path.dirname(cartella_interfaccia)
if cartella_progetto not in sys.path:
    sys.path.insert(0, cartella_progetto)

import streamlit as st
from dotenv import load_dotenv

import config as cfg
from Sorgente.Pipeline import Pipeline

load_dotenv()

st.set_page_config(
    page_title='Fact Checker',
    page_icon='🔍',
    layout='wide'
)

@st.cache_resource
def carica_pipeline(api_key: str) -> Pipeline:
    return Pipeline(api_key=api_key)

def colore_verdetto(verdetto: str) -> str:
    return {
        'SUPPORTS': '#2ecc71',
        'REFUTES': '#e74c3c',
        'NOT ENOUGH INFO': '#f39c12'
    }.get(verdetto, '#95a5a6')


def icona_verdetto(verdetto: str) -> str:
    return {
        'SUPPORTS': '✅',
        'REFUTES': '❌',
        'NOT ENOUGH INFO': '⚠️'
    }.get(verdetto, '❓')

def main():
    st.title('🔍 Fact Checker')
    st.markdown(
        'Sistema automatico di verifica delle affermazioni '
        'basato su **BM25 esteso** + **NLI con LLM**.'
    )
    st.divider()

    with st.sidebar:
        st.header('⚙️ Configurazione')
        api_key = os.getenv('OPENAI_API_KEY')
        st.subheader('Sorgente')
        sorgente = st.radio(
            'Corpus da usare',
            options=['fever', 'web'],
            format_func=lambda x: (
                'FEVER (corpus fisso)' if x == 'fever' else 'Wikipedia live'
            ),
            help=(
                'FEVER: valutazione rigorosa su corpus pre-costruito.\n'
                'Web: risponde a qualsiasi domanda in tempo reale.'
            )
        )
        cfg.SORGENTE_CORPUS = sorgente
        st.subheader('Parametri')
        k_documenti = st.slider(
            'Documenti da recuperare (K)',
            min_value=1, max_value=10,
            value=cfg.DOCUMENTI_DA_RECUPERARE,
            help='Quanti documenti usare per la classificazione NLI'
        )
        cfg.DOCUMENTI_DA_RECUPERARE = k_documenti
        usa_espansione = st.toggle(
            'Espansione query (WordNet)',
            value=cfg.ESPANSIONE_ATTIVA,
            help=(
                'Aggiunge sinonimi WordNet ai termini della query.\n'
                'Impatto marginale su FEVER — utile per query generali.'
            )
        )
        cfg.ESPANSIONE_ATTIVA = usa_espansione

        st.divider()
        st.markdown('**Parametri BM25:**')
        st.caption(f'k1 = {cfg.K1} · b = {cfg.B}')
        st.caption(f'α (authority) = {cfg.PESO_AUTHORITY}')
        st.caption(f'β (freshness) = {cfg.PESO_FRESHNESS}')
        st.caption(f'Modello LLM: {cfg.MODELLO_LLM}')
    if not api_key:
        st.info('Imposta la variabile d\'ambiente OPENAI_API_KEY per iniziare.')
        return
    try:
        with st.spinner('Caricamento sistema...'):
            pipeline = carica_pipeline(api_key)
    except FileNotFoundError as e:
        st.error(
            f'File non trovato: {e}\n\n'
            'Esegui prima i notebook di setup (00 e 02).'
        )
        return
    st.subheader('📝 Inserisci un\'affermazione da verificare')

    esempi = [
        'Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.',
        'The Eiffel Tower is located in Berlin.',
        'Barack Obama was born in Hawaii.',
        'Cellulari vietati nelle scuole italiane.',
    ]
    esempio_scelto = st.selectbox(
        'Oppure scegli un esempio:',
        [''] + esempi,
        format_func=lambda x: x if x else '— seleziona un esempio —'
    )

    claim_input = st.text_area(
        'Affermazione:',
        value=esempio_scelto,
        height=80,
        placeholder='Scrivi qui l\'affermazione da verificare...'
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        verifica_btn = st.button('🔍 Verifica', type='primary', use_container_width=True)
    with col2:
        if usa_espansione and claim_input and sorgente == 'fever':
            try:
                stats = pipeline._espansore.statistiche_espansione(claim_input)
                if stats['n_aggiunti'] > 0:
                    st.caption(
                        f"Espansione: +{stats['n_aggiunti']} sinonimi "
                        f"→ {stats['sinonimi_aggiunti']}"
                    )
            except Exception:
                pass

    if verifica_btn and claim_input.strip():
        with st.spinner('Verifica in corso...'):
            try:
                risultato = pipeline.verifica(claim_input.strip())
            except Exception as e:
                st.error(f'Errore durante la verifica: {e}')
                return

        verdetto   = risultato['verdetto']
        confidenza = risultato['confidenza']
        colore     = colore_verdetto(verdetto)
        icona      = icona_verdetto(verdetto)

        st.divider()
        st.subheader('📊 Risultato')

        # Verdetto principale
        st.markdown(
            f'<div style="background:{colore}22; border-left:5px solid {colore}; '
            f'padding:16px; border-radius:6px; margin-bottom:16px;">'
            f'<h2 style="color:{colore}; margin:0;">{icona} {verdetto}</h2>'
            f'<p style="margin:4px 0 0 0; color:gray;">Confidenza: {confidenza:.1f}%</p>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Token query (con espansione se attiva)
        token_query = risultato.get('token_query', [])
        if token_query and usa_espansione:
            st.caption(f'Token query usati: {" · ".join(token_query)}')

        # Pesi per label
        pesi = risultato.get('pesi_label', {})
        if pesi:
            col_s, col_r, col_n = st.columns(3)
            with col_s:
                st.metric('SUPPORTS',        f"{pesi.get('SUPPORTS', 0):.2f}")
            with col_r:
                st.metric('REFUTES',         f"{pesi.get('REFUTES', 0):.2f}")
            with col_n:
                st.metric('NOT ENOUGH INFO', f"{pesi.get('NOT ENOUGH INFO', 0):.2f}")

        # Documenti analizzati
        st.divider()
        st.subheader('📄 Documenti analizzati')

        documenti = risultato.get('documenti', [])
        if documenti:
            for i, doc in enumerate(documenti, 1):
                col_doc   = colore_verdetto(doc['verdetto'])
                icona_doc = icona_verdetto(doc['verdetto'])

                with st.expander(
                    f"{i}. {doc['titolo']} — {icona_doc} {doc['verdetto']} "
                    f"(score: {doc['punteggio']:.3f})"
                ):
                    st.markdown(f"**Verdetto LLM:** {icona_doc} {doc['verdetto']}")
                    st.markdown(f"**Punteggio BM25 esteso:** {doc['punteggio']:.4f}")
                    st.markdown('**Testo evidenza usato:**')
                    st.info(
                        doc['evidenza'] if doc['evidenza']
                        else '*(nessun testo disponibile)*'
                    )
        else:
            st.warning('Nessun documento recuperato per questa query.')

    elif verifica_btn and not claim_input.strip():
        st.warning('Inserisci un\'affermazione prima di premere Verifica.')


if __name__ == '__main__':
    main()