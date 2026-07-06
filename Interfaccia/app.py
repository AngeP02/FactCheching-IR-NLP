import os
import sys
import pickle
import streamlit as st
import config as cfg
from Sorgente.Preprocessing.preprocesing import Preprocessore
from Sorgente.Indicizzazione.indice_invertito import IndiceInvertito
from Sorgente.Recupero.metadati import GestoreMetadati
from Sorgente.Recupero.bm25 import CalcolatoreRanking
from Sorgente.Classificazione.classificatore_nli import ClassificatoreNLI
from Sorgente.Espansione.espansione_query import EspansoreQuery
from dotenv import load_dotenv

load_dotenv()
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

st.set_page_config(
    page_title = 'Fact Checker',
    page_icon  = '🔍',
    layout     = 'wide'
)

@st.cache_resource
def carica_risorse(api_key: str):
    """
    Carica tutte le risorse del sistema e le mette in cache.
    """
    with open(cfg.PERCORSO_CORPUS_ARRICCHITO, 'rb') as f:
        corpus = pickle.load(f)

    indice = IndiceInvertito()
    indice.carica(cfg.PERCORSO_INDICE)

    preprocessore = Preprocessore()
    metadati = GestoreMetadati(corpus)
    ranking = CalcolatoreRanking(indice, preprocessore, metadati)
    classificatore = ClassificatoreNLI(api_key)
    espansore = EspansoreQuery(preprocessore)

    return {
        'corpus': corpus,
        'indice': indice,
        'preprocessore': preprocessore,
        'ranking': ranking,
        'classificatore': classificatore,
        'espansore': espansore
    }


def colore_verdetto(verdetto: str) -> str:
    """Restituisce il colore associato a ogni label."""
    return {
        'SUPPORTS':        '#2ecc71',
        'REFUTES':         '#e74c3c',
        'NOT ENOUGH INFO': '#f39c12'
    }.get(verdetto, '#95a5a6')


def icona_verdetto(verdetto: str) -> str:
    """Restituisce l'icona associata a ogni label."""
    return {
        'SUPPORTS':        '✅',
        'REFUTES':         '❌',
        'NOT ENOUGH INFO': '⚠️'
    }.get(verdetto, '❓')


def esegui_verifica(
    claim:         str,
    risorse:       dict,
    k_documenti:   int,
    usa_espansione: bool,
    sorgente:      str
) -> dict:
    """
    Esegue la verifica della claim attraverso la pipeline.

    Args:
        claim:          testo della claim
        risorse:        dizionario risorse caricate
        k_documenti:    numero documenti da recuperare
        usa_espansione: se True usa WordNet
        sorgente:       'fever' o 'web'

    Returns:
        Dizionario risultato della pipeline
    """
    ranking       = risorse['ranking']
    classificatore = risorse['classificatore']
    corpus        = risorse['corpus']
    espansore     = risorse['espansore']

    if sorgente == 'web':
        from Sorgente.Corpus.Corpus_web import CorpusWeb
        from Sorgente.Indicizzazione.indice_invertito import IndiceInvertito
        from Sorgente.Recupero.metadati import GestoreMetadati
        from Sorgente.Recupero.bm25 import CalcolatoreRanking

        corpus_web     = CorpusWeb(risorse['preprocessore'], max_pagine=k_documenti)
        corpus_runtime = corpus_web.ottieni_documenti(claim)

        if not corpus_runtime:
            return {
                'claim': claim, 'verdetto': 'NOT ENOUGH INFO',
                'confidenza': 0.0, 'pesi_label': {}, 'documenti': []
            }

        indice_temp   = IndiceInvertito()
        for id_doc, doc in corpus_runtime.items():
            token = risorse['preprocessore'].elabora(doc.get('text', ''))
            indice_temp.aggiungi_documento(id_doc, token)

        metadati_temp = GestoreMetadati(corpus_runtime)
        ranking_temp  = CalcolatoreRanking(
            indice_temp, risorse['preprocessore'], metadati_temp
        )
        risultati_bm25 = ranking_temp.recupera_top_k(claim, k=k_documenti)
        corpus_da_usare = corpus_runtime

    else:
        if usa_espansione:
            token_espansi = espansore.espandi(claim)
            query_espansa = ' '.join(token_espansi)
            risultati_bm25 = ranking.recupera_top_k(query_espansa, k=k_documenti)
        else:
            risultati_bm25 = ranking.recupera_top_k(claim, k=k_documenti)
        corpus_da_usare = corpus

    if not risultati_bm25:
        return {
            'claim': claim, 'verdetto': 'NOT ENOUGH INFO',
            'confidenza': 0.0, 'pesi_label': {}, 'documenti': []
        }

    risultato_nli = classificatore.verifica(
        claim             = claim,
        risultati_bm25    = risultati_bm25,
        corpus_arricchito = corpus_da_usare
    )

    return {
        'claim':      claim,
        'verdetto':   risultato_nli['verdetto'],
        'confidenza': risultato_nli['confidenza'],
        'pesi_label': risultato_nli['pesi_label'],
        'documenti':  risultato_nli['dettagli']
    }


# ── Interfaccia ───────────────────────────────────────────────────

def main():
    # Titolo
    st.title('🔍 Fact Checker')
    st.markdown(
        'Sistema automatico di verifica delle affermazioni '
        'basato su **BM25** + **NLI con LLM**.'
    )
    st.divider()

    # ── Sidebar ──────────────────────────────────────────────────
    with st.sidebar:
        st.header('⚙️ Configurazione')

        api_key = os.getenv('OPENAI_API_KEY')

        st.subheader('Sorgente')
        sorgente = st.radio(
            'Corpus da usare',
            options    = ['fever', 'web'],
            format_func = lambda x: '📚 FEVER (corpus fisso)' if x == 'fever'
                                    else '🌐 Wikipedia live',
            help = ('FEVER: valutazione rigorosa su corpus pre-costruito.\n'
                    'Web: risponde a qualsiasi domanda in tempo reale.')
        )

        st.subheader('Parametri')
        k_documenti = st.slider(
            'Documenti da recuperare (K)',
            min_value = 1, max_value = 10,
            value     = cfg.DOCUMENTI_DA_RECUPERARE,
            help      = 'Quanti documenti usare per la classificazione NLI'
        )

        usa_espansione = st.toggle(
            'Espansione query (WordNet)',
            value = cfg.ESPANSIONE_ATTIVA,
            help  = ('Aggiunge sinonimi WordNet ai termini della query.\n'
                     'Impatto marginale su FEVER — utile per query generali.')
        )

        st.divider()
        st.markdown('**Parametri BM25:**')
        st.caption(f'k1 = {cfg.K1} · b = {cfg.B}')
        st.caption(f'α (authority) = {cfg.PESO_AUTHORITY}')
        st.caption(f'β (freshness) = {cfg.PESO_FRESHNESS}')
        st.caption(f'Modello LLM: {cfg.MODELLO_LLM}')

    # ── Area principale ───────────────────────────────────────────
    if not api_key:
        st.info('👈 Inserisci la tua OpenAI API Key nella sidebar per iniziare.')
        return

    # Carica le risorse (con cache)
    try:
        with st.spinner('Caricamento sistema...'):
            risorse = carica_risorse(api_key)
    except FileNotFoundError as e:
        st.error(f'File non trovato: {e}\n\nEsegui prima i notebook di setup.')
        return

    # Input claim
    st.subheader('📝 Inserisci un\'affermazione da verificare')

    esempi = [
        'Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.',
        'The Eiffel Tower is located in Berlin.',
        'Barack Obama was born in Hawaii.',
        'Cellulari vietati nelle scuole italiane.'
    ]
    esempio_scelto = st.selectbox(
        'Oppure scegli un esempio:',
        [''] + esempi,
        format_func = lambda x: x if x else '— seleziona un esempio —'
    )

    claim_input = st.text_area(
        'Affermazione:',
        value       = esempio_scelto,
        height      = 80,
        placeholder = 'Scrivi qui l\'affermazione da verificare...'
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        verifica_btn = st.button('🔍 Verifica', type='primary', use_container_width=True)
    with col2:
        if usa_espansione and claim_input:
            token_esp = risorse['espansore'].statistiche_espansione(claim_input)
            if token_esp['n_aggiunti'] > 0:
                st.caption(
                    f"Espansione: +{token_esp['n_aggiunti']} sinonimi "
                    f"→ {token_esp['sinonimi_aggiunti']}"
                )

    # ── Risultati ─────────────────────────────────────────────────
    if verifica_btn and claim_input.strip():
        with st.spinner('Verifica in corso...'):
            try:
                risultato = esegui_verifica(
                    claim_input.strip(),
                    risorse,
                    k_documenti,
                    usa_espansione,
                    sorgente
                )
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

        # Pesi per label
        pesi = risultato['pesi_label']
        if pesi:
            col_s, col_r, col_n = st.columns(3)
            peso_tot = sum(pesi.values()) or 1
            with col_s:
                st.metric('SUPPORTS',        f"{pesi.get('SUPPORTS', 0):.2f}")
            with col_r:
                st.metric('REFUTES',         f"{pesi.get('REFUTES', 0):.2f}")
            with col_n:
                st.metric('NOT ENOUGH INFO', f"{pesi.get('NOT ENOUGH INFO', 0):.2f}")

        # Dettagli documenti
        st.divider()
        st.subheader('📄 Documenti analizzati')

        if risultato['documenti']:
            for i, doc in enumerate(risultato['documenti'], 1):
                col_doc = colore_verdetto(doc['verdetto'])
                icona_doc = icona_verdetto(doc['verdetto'])

                with st.expander(
                    f"{i}. {doc['titolo']} — {icona_doc} {doc['verdetto']} "
                    f"(score: {doc['punteggio']:.3f})"
                ):
                    st.markdown(f"**Verdetto LLM:** {icona_doc} {doc['verdetto']}")
                    st.markdown(f"**Punteggio BM25 esteso:** {doc['punteggio']:.4f}")
                    st.markdown('**Testo evidenza usato:**')
                    st.info(doc['evidenza'] if doc['evidenza'] else '*(nessun testo disponibile)*')
        else:
            st.warning('Nessun documento recuperato per questa query.')

    elif verifica_btn and not claim_input.strip():
        st.warning('Inserisci un\'affermazione prima di premere Verifica.')


if __name__ == '__main__':
    main()