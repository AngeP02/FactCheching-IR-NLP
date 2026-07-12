import os
import sys
import json
import datetime

# ── Path setup ────────────────────────────────────────────────────
cartella_interfaccia = os.path.dirname(os.path.abspath(__file__))
cartella_progetto    = os.path.dirname(cartella_interfaccia)
if cartella_progetto not in sys.path:
    sys.path.insert(0, cartella_progetto)

import streamlit as st
from dotenv import load_dotenv
import config as cfg
from Sorgente.Pipeline import Pipeline

load_dotenv()

# ── Configurazione pagina ─────────────────────────────────────────
st.set_page_config(
    page_title='Fact Checker AI',
    page_icon='🔍',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# ── CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Nascondi sidebar e menu */
    [data-testid="stSidebar"]      { display: none; }
    [data-testid="collapsedControl"]{ display: none; }
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }

    /* Sfondo generale */
    .stApp { background-color: #FAFAFA; }

    /* Navigazione superiore */
    .nav-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 0 10px 0;
        border-bottom: 1px solid #E8E8E8;
        margin-bottom: 32px;
    }
    .nav-logo {
        font-size: 1.2em;
        font-weight: 800;
        color: #1A1A2E;
        letter-spacing: -0.5px;
    }
    .nav-logo span { color: #2563EB; }
    .nav-links {
        display: flex;
        gap: 28px;
    }
    .nav-link {
        font-size: 0.85em;
        font-weight: 600;
        color: #6B7280;
        text-decoration: none;
        cursor: pointer;
        padding: 4px 0;
        border-bottom: 2px solid transparent;
    }
    .nav-link.active {
        color: #1A1A2E;
        border-bottom: 2px solid #2563EB;
    }

    /* Barra di ricerca hero */
    .hero {
        text-align: center;
        padding: 48px 0 36px 0;
    }
    .hero-title {
        font-size: 2.6em;
        font-weight: 800;
        color: #1A1A2E;
        line-height: 1.15;
        margin-bottom: 10px;
        letter-spacing: -1px;
    }
    .hero-sub {
        font-size: 1.05em;
        color: #6B7280;
        margin-bottom: 32px;
    }

    /* Verdetti badge */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 16px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.9em;
        letter-spacing: 0.3px;
    }
    .badge-supports { background: #DCFCE7; color: #166534; }
    .badge-refutes  { background: #FEE2E2; color: #991B1B; }
    .badge-nei      { background: #FEF3C7; color: #92400E; }

    /* Card verdetto risultato */
    .verdetto-card {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .verdetto-claim {
        font-size: 1.05em;
        color: #374151;
        font-style: italic;
        margin-bottom: 16px;
        line-height: 1.5;
    }
    .verdetto-label {
        font-size: 0.7em;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #9CA3AF;
        margin-bottom: 6px;
    }
    .spiegazione-box {
        background: #F8FAFF;
        border-left: 3px solid #2563EB;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        color: #374151;
        font-size: 0.88em;
        line-height: 1.6;
        margin-top: 14px;
    }

    /* Confidenza */
    .conf-label {
        font-size: 0.72em;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9CA3AF;
        margin-bottom: 4px;
    }
    .conf-value {
        font-size: 2em;
        font-weight: 800;
        color: #1A1A2E;
        line-height: 1;
    }

    /* Card trend */
    .trend-card {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 18px 20px;
        height: 100%;
        transition: box-shadow 0.15s;
        cursor: pointer;
    }
    .trend-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
    .trend-data {
        font-size: 0.7em;
        color: #9CA3AF;
        margin-bottom: 8px;
        font-weight: 500;
    }
    .trend-claim {
        font-size: 0.88em;
        color: #1A1A2E;
        font-weight: 600;
        line-height: 1.4;
        margin-bottom: 10px;
    }

    /* Documenti */
    .doc-card {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 8px;
    }
    .doc-title {
        font-weight: 700;
        color: #1A1A2E;
        font-size: 0.88em;
        margin-bottom: 3px;
    }
    .doc-meta {
        font-family: monospace;
        font-size: 0.72em;
        color: #9CA3AF;
        margin-bottom: 6px;
    }
    .doc-ev {
        font-size: 0.82em;
        color: #6B7280;
        line-height: 1.5;
        border-left: 2px solid #E5E7EB;
        padding-left: 10px;
    }

    /* Chip token */
    .chip {
        display: inline-block;
        background: #F3F4F6;
        color: #374151;
        font-family: monospace;
        font-size: 0.73em;
        padding: 2px 8px;
        border-radius: 4px;
        margin: 2px;
    }
    .chip-blue {
        display: inline-block;
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        color: #2563EB;
        font-family: monospace;
        font-size: 0.73em;
        padding: 2px 8px;
        border-radius: 4px;
        margin: 2px;
    }

    /* Metriche box */
    .mbox {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .mbox-val {
        font-size: 1.9em;
        font-weight: 800;
        color: #2563EB;
        font-family: monospace;
        line-height: 1;
        margin-bottom: 4px;
    }
    .mbox-label {
        font-size: 0.68em;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9CA3AF;
    }

    /* Sezione label */
    .slabel {
        font-size: 0.68em;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #2563EB;
        margin-bottom: 12px;
        margin-top: 4px;
    }

    /* Tabs override */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 2px solid #E5E7EB;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.85em;
        font-weight: 600;
        color: #6B7280;
        padding: 10px 20px;
        border-radius: 0;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #1A1A2E !important;
        border-bottom: 2px solid #2563EB;
    }
</style>
""", unsafe_allow_html=True)


# ── Percorso file trend ───────────────────────────────────────────
PERCORSO_TREND = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'trend_verifiche.json'
)


# ── Gestione trend (salvataggio su file JSON) ─────────────────────

def carica_trend() -> list[dict]:
    """Carica le ultime verifiche salvate su file."""
    if not os.path.exists(PERCORSO_TREND):
        return []
    try:
        with open(PERCORSO_TREND, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []


def salva_trend(claim: str, verdetto: str, confidenza: float):
    """Aggiunge una nuova verifica al feed trend (max 12 voci)."""
    trend = carica_trend()
    nuova = {
        'claim':      claim,
        'verdetto':   verdetto,
        'confidenza': confidenza,
        'data':       datetime.datetime.now().strftime('%d/%m/%Y %H:%M')
    }
    # Evita duplicati consecutivi
    if trend and trend[0]['claim'] == claim:
        return
    trend.insert(0, nuova)
    trend = trend[:12]
    try:
        with open(PERCORSO_TREND, 'w', encoding='utf-8') as f:
            json.dump(trend, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ── Cache pipeline ────────────────────────────────────────────────

@st.cache_resource
def carica_pipeline(api_key: str) -> Pipeline:
    return Pipeline(api_key=api_key)


# ── Helpers UI ────────────────────────────────────────────────────

def icona(v: str) -> str:
    return {'SUPPORTS': '✅', 'REFUTES': '❌', 'NOT ENOUGH INFO': '⚠️'}.get(v, '❓')

def colore_verdetto(v: str) -> str:
    return {
        'SUPPORTS':       '#166534',
        'REFUTES':        '#991B1B',
        'NOT ENOUGH INFO':'#92400E'
    }.get(v, '#374151')

def classe_badge(v: str) -> str:
    return {
        'SUPPORTS':       'badge badge-supports',
        'REFUTES':        'badge badge-refutes',
        'NOT ENOUGH INFO':'badge badge-nei'
    }.get(v, 'badge badge-nei')

def badge_html(v: str) -> str:
    return f'<span class="{classe_badge(v)}">{icona(v)} {v}</span>'

def mbox(valore: str, label: str):
    st.markdown(
        f'<div class="mbox">'
        f'<div class="mbox-val">{valore}</div>'
        f'<div class="mbox-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

def render_doc(doc: dict, rank: int):
    titolo = doc.get('titolo', doc.get('id_doc', ''))
    meta   = f"BM25: {doc.get('punteggio', 0):.4f}"
    if 'score_reranking' in doc:
        meta += f"  ·  Re-rank: {doc['score_reranking']:.4f}"
    meta += f"  ·  {icona(doc['verdetto'])} {doc['verdetto']}"
    ev = doc.get('evidenza', '')
    ev_html = (
        f'<div class="doc-ev">{ev[:350]}{"…" if len(ev) > 350 else ""}</div>'
        if ev else ''
    )
    st.markdown(
        f'<div class="doc-card">'
        f'<div class="doc-title">{rank}. {titolo}</div>'
        f'<div class="doc-meta">{meta}</div>'
        f'{ev_html}'
        f'</div>',
        unsafe_allow_html=True
    )


def init_state():
    defaults = {
        'pagina':      'home',
        'last_result': None,
        'last_stats':  None,
        'last_claim':  '',
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Navigazione ───────────────────────────────────────────────────

def navbar():
    """Barra di navigazione superiore con link alle pagine."""
    pagine = {
        'home':        '🏠 Home',
        'sviluppatori':'⚙️ Analisi Tecnica',
        'benchmark':   '📊 Benchmark',
        'impostazioni':'🔧 Impostazioni',
    }
    cols = st.columns([2, 1, 1, 1, 1])
    with cols[0]:
        st.markdown(
            '<div class="nav-logo">Fact<span>Check</span> AI</div>',
            unsafe_allow_html=True
        )
    for col, (chiave, etichetta) in zip(cols[1:], pagine.items()):
        with col:
            attiva = 'primary' if st.session_state.pagina == chiave else 'secondary'
            if st.button(etichetta, key=f'nav_{chiave}', type=attiva, use_container_width=True):
                st.session_state.pagina = chiave
                st.rerun()
    st.divider()


# ── Pagina Home ───────────────────────────────────────────────────

def pagina_home(pipeline: Pipeline):

    # Hero + barra di ricerca
    st.markdown(
        '<div class="hero">'
        '<div class="hero-title">Verifica i fatti,<br>non le opinioni.</div>'
        '<div class="hero-sub">Inserisci un\'affermazione: il sistema cercherà le evidenze e ti darà un verdetto.</div>'
        '</div>',
        unsafe_allow_html=True
    )

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        claim = st.text_input(
            label='claim',
            label_visibility='collapsed',
            placeholder='Es: The Eiffel Tower is located in Paris, France.',
            value=st.session_state.last_claim,
            key='input_claim'
        )
    with col_btn:
        verifica = st.button(
            '🔍 Verifica',
            type='primary',
            use_container_width=True
        )

    # Impostazioni rapide inline
    with st.expander('⚙️ Impostazioni rapide', expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            cfg.DOCUMENTI_DA_RECUPERARE = st.slider(
                'Documenti Top-K', 1, 10, cfg.DOCUMENTI_DA_RECUPERARE
            )
        with c2:
            if hasattr(cfg, 'RERANKING_ATTIVO'):
                cfg.RERANKING_ATTIVO = st.toggle(
                    '🔁 Re-ranking neurale',
                    value=cfg.RERANKING_ATTIVO
                )
        with c3:
            cfg.ESPANSIONE_ATTIVA = st.toggle(
                '🔤 Espansione WordNet',
                value=cfg.ESPANSIONE_ATTIVA
            )

    # Verifica
    if verifica and claim.strip():
        st.session_state.last_claim = claim
        with st.spinner('Recupero evidenze e classificazione in corso…'):
            try:
                stats = None
                if cfg.ESPANSIONE_ATTIVA:
                    try:
                        stats = pipeline._espansore.statistiche_espansione(claim)
                    except Exception:
                        pass

                risultato = pipeline.verifica(claim)
                st.session_state.last_result = risultato
                st.session_state.last_stats  = stats

                salva_trend(claim, risultato['verdetto'], risultato['confidenza'])

            except Exception as e:
                st.error(f'Errore durante la verifica: {e}')
                return

    # Mostra risultato
    risultato = st.session_state.last_result
    if risultato and risultato['claim'] == st.session_state.last_claim:
        _mostra_risultato(risultato, st.session_state.last_stats)

    # Feed trend
    st.markdown('<br>', unsafe_allow_html=True)
    _mostra_trend_feed()


def _mostra_risultato(risultato: dict, stats):
    """Rende il pannello verdetto nella Home."""
    verdetto    = risultato['verdetto']
    confidenza  = risultato['confidenza']
    spiegazione = risultato.get('spiegazione', '')
    documenti   = risultato.get('documenti', [])
    pesi        = risultato.get('pesi_label', {})
    token       = risultato.get('token_query', [])

    st.markdown('<br>', unsafe_allow_html=True)

    col_sinistra, col_destra = st.columns([3, 2], gap='large')

    with col_sinistra:
        st.markdown(
            f'<div class="verdetto-card">'
            f'<div class="verdetto-label">Claim verificata</div>'
            f'<div class="verdetto-claim">"{risultato["claim"]}"</div>'
            f'{badge_html(verdetto)}'
            f'{"<div class=\'spiegazione-box\'>" + spiegazione + "</div>" if spiegazione else ""}'
            f'</div>',
            unsafe_allow_html=True
        )

        # Barre pesi label
        if pesi:
            st.markdown('<div class="slabel">Distribuzione confidenza</div>', unsafe_allow_html=True)
            totale = max(sum(pesi.values()), 0.001)
            colori = {
                'SUPPORTS':       '#16A34A',
                'REFUTES':        '#DC2626',
                'NOT ENOUGH INFO':'#D97706'
            }
            for label, colore in colori.items():
                val = pesi.get(label, 0)
                pct = min(val / totale * 100, 100)
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">'
                    f'<span style="width:140px;font-size:0.75em;color:#6B7280;font-weight:600;">{label}</span>'
                    f'<div style="flex:1;height:8px;background:#F3F4F6;border-radius:4px;overflow:hidden;">'
                    f'<div style="width:{pct:.1f}%;height:100%;background:{colore};border-radius:4px;"></div></div>'
                    f'<span style="font-family:monospace;font-size:0.73em;color:#9CA3AF;width:50px;text-align:right;">{val:.2f}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    with col_destra:
        # Confidenza grande
        st.markdown(
            f'<div class="mbox" style="margin-bottom:16px;">'
            f'<div class="conf-label">Confidenza del sistema</div>'
            f'<div class="conf-value">{confidenza:.0f}%</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.progress(int(confidenza) / 100)

        # Token query
        if token:
            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown('<div class="slabel">Token query</div>', unsafe_allow_html=True)
            st.markdown(
                ''.join(f'<span class="chip">{t}</span>' for t in token),
                unsafe_allow_html=True
            )

        # WordNet
        if stats and stats.get('n_aggiunti', 0) > 0:
            st.markdown('<div class="slabel" style="margin-top:12px;">Espansione WordNet</div>', unsafe_allow_html=True)
            orig    = ''.join(f'<span class="chip">{t}</span>' for t in stats.get('token_originali', []))
            aggiunti= ''.join(f'<span class="chip-blue">{t}</span>' for t in stats.get('sinonimi_aggiunti', []))
            st.markdown(f'Originali: {orig}<br>Aggiunti: {aggiunti}', unsafe_allow_html=True)

    # Documenti recuperati
    if documenti:
        st.markdown('<br>', unsafe_allow_html=True)
        with st.expander(f'📚 {len(documenti)} fonti analizzate', expanded=False):
            for i, doc in enumerate(documenti, 1):
                render_doc(doc, i)


def _mostra_trend_feed():
    """Griglia di card con le ultime verifiche effettuate."""
    trend = carica_trend()
    if not trend:
        return

    st.markdown('<div class="slabel">Ultime verifiche</div>', unsafe_allow_html=True)

    # Griglia 3 colonne
    righe = [trend[i:i+3] for i in range(0, min(len(trend), 9), 3)]
    for riga in righe:
        cols = st.columns(3, gap='medium')
        for col, voce in zip(cols, riga):
            with col:
                st.markdown(
                    f'<div class="trend-card">'
                    f'<div class="trend-data">{voce["data"]}</div>'
                    f'<div class="trend-claim">{voce["claim"][:120]}{"…" if len(voce["claim"]) > 120 else ""}</div>'
                    f'{badge_html(voce["verdetto"])}'
                    f'<span style="font-size:0.75em;color:#9CA3AF;margin-left:8px;">{voce["confidenza"]:.0f}% confidenza</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )


# ── Pagina Analisi Tecnica ────────────────────────────────────────

def pagina_sviluppatori():
    st.title('⚙️ Analisi Tecnica')
    st.caption("Dettagli IR e NLP dell'ultima query elaborata.")

    risultato = st.session_state.last_result
    if not risultato:
        st.info("Esegui prima una verifica dalla Home.")
        return

    verdetto    = risultato['verdetto']
    confidenza  = risultato['confidenza']
    pesi        = risultato.get('pesi_label', {})
    spiegazione = risultato.get('spiegazione', '')
    documenti   = risultato.get('documenti', [])
    token       = risultato.get('token_query', [])
    stats       = st.session_state.last_stats

    # Header
    st.markdown(
        f'{badge_html(verdetto)}'
        f'<span style="color:#6B7280;font-size:0.9em;margin-left:12px;">Confidenza: {confidenza:.1f}%</span>'
        f'<br><span style="color:#9CA3AF;font-size:0.8em;font-style:italic;">"{risultato["claim"][:100]}{"…" if len(risultato["claim"]) > 100 else ""}"</span>',
        unsafe_allow_html=True
    )
    st.markdown('<br>', unsafe_allow_html=True)

    tab_nlp, tab_ir = st.tabs(['🧠 NLP · Classificazione', '📄 IR · Documenti'])

    with tab_nlp:
        st.markdown('<br>', unsafe_allow_html=True)
        col_nlp, col_conf = st.columns([2, 1], gap='large')

        with col_nlp:
            # Barre pesi
            if pesi:
                st.markdown('<div class="slabel">Pesi per label</div>', unsafe_allow_html=True)
                totale = max(sum(pesi.values()), 0.001)
                colori = {
                    'SUPPORTS':       '#16A34A',
                    'REFUTES':        '#DC2626',
                    'NOT ENOUGH INFO':'#D97706'
                }
                for label, colore in colori.items():
                    val = pesi.get(label, 0)
                    pct = min(val / totale * 100, 100)
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">'
                        f'<span style="width:145px;font-size:0.75em;color:#6B7280;font-weight:600;">{label}</span>'
                        f'<div style="flex:1;height:8px;background:#F3F4F6;border-radius:4px;overflow:hidden;">'
                        f'<div style="width:{pct:.1f}%;height:100%;background:{colore};border-radius:4px;"></div></div>'
                        f'<span style="font-family:monospace;font-size:0.73em;color:#9CA3AF;width:50px;text-align:right;">{val:.2f}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            # Spiegazione
            if spiegazione:
                st.markdown('<br>', unsafe_allow_html=True)
                st.markdown('<div class="slabel">Spiegazione LLM</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="spiegazione-box">{spiegazione}</div>', unsafe_allow_html=True)

            # Token
            if token:
                st.markdown('<br>', unsafe_allow_html=True)
                st.markdown('<div class="slabel">Token query</div>', unsafe_allow_html=True)
                st.markdown(
                    ''.join(f'<span class="chip">{t}</span>' for t in token),
                    unsafe_allow_html=True
                )

            # WordNet
            if stats and stats.get('n_aggiunti', 0) > 0:
                st.markdown('<br>', unsafe_allow_html=True)
                st.markdown('<div class="slabel">Espansione WordNet</div>', unsafe_allow_html=True)
                orig    = ''.join(f'<span class="chip">{t}</span>' for t in stats.get('token_originali', []))
                aggiunti= ''.join(f'<span class="chip-blue">{t}</span>' for t in stats.get('sinonimi_aggiunti', []))
                st.markdown(f'Originali: {orig}<br>Aggiunti: {aggiunti}', unsafe_allow_html=True)

        with col_conf:
            st.markdown(
                f'<div class="mbox">'
                f'<div class="conf-label">Confidenza</div>'
                f'<div class="conf-value">{confidenza:.0f}%</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.progress(int(confidenza) / 100)

    with tab_ir:
        st.markdown('<br>', unsafe_allow_html=True)
        if documenti:
            for i, doc in enumerate(documenti, 1):
                render_doc(doc, i)
        else:
            st.info('Nessun documento recuperato.')


# ── Pagina Benchmark ──────────────────────────────────────────────

def pagina_benchmark(pipeline: Pipeline):
    st.title('📊 Benchmark')
    st.caption("Valuta il sistema su un campione del dataset FEVER.")

    st.info(
        "Il benchmark carica automaticamente il dataset FEVER da HuggingFace "
        "(già in cache se hai eseguito i notebook). "
        "Non è necessario nessun file esterno."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        n_per_classe = st.slider(
            'Claim per classe (SUPPORTS / REFUTES / NOT ENOUGH INFO)',
            min_value=2, max_value=50, value=10,
            help='Totale = n × 3. Es: 10 → 30 claim totali.'
        )
    with col_b:
        split_scelto = st.selectbox(
            'Split FEVER',
            ['labelled_dev', 'paper_dev'],
            help='labelled_dev = sviluppo, paper_dev = test finale'
        )

    st.caption(f"Claim totali: **{n_per_classe * 3}** ({n_per_classe} per classe)")

    if not st.button('🚀 Avvia Benchmark', type='primary'):
        return

    # Carica dataset
    with st.spinner('Caricamento dataset FEVER…'):
        try:
            from datasets import load_dataset
            import pandas as pd

            ds       = load_dataset('fever', 'v1.0')
            df_raw   = pd.DataFrame(ds[split_scelto])
            df       = df_raw.groupby('id').agg(
                claim         =('claim',               'first'),
                label         =('label',               'first'),
                url_documenti =('evidence_wiki_url',   lambda x: list(x.dropna().unique())),
                sentence_ids  =('evidence_sentence_id', list)
            ).reset_index()

            campioni = []
            for lbl in ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']:
                sub = df[df['label'] == lbl]
                campioni.append(sub.sample(n=min(n_per_classe, len(sub)), random_state=42))
            df_campione = pd.concat(campioni).reset_index(drop=True)

            st.success(f"Dataset caricato: {len(df_campione)} claim ({n_per_classe} per classe)")

        except Exception as e:
            st.error(f"Errore nel caricamento del dataset: {e}")
            return

    # Valutazione
    lista_ir   = []
    label_vere = []
    label_pred = []

    barra  = st.progress(0, text='Valutazione claim…')
    totale = len(df_campione)

    for i, (_, riga) in enumerate(df_campione.iterrows()):
        try:
            sids_map = {}
            for url, sid in zip(riga['url_documenti'], riga['sentence_ids']):
                if url:
                    sids_map.setdefault(url, []).append(sid)

            ris  = pipeline.verifica(riga['claim'], sentence_ids_map=sids_map or None)
            docs = [d['id_doc'] for d in ris.get('documenti', [])]
            rilevanti = set(u for u in riga['url_documenti'] if u)

            if rilevanti:
                lista_ir.append({'recuperati': docs, 'rilevanti': rilevanti})

            label_vere.append(riga['label'])
            label_pred.append(ris.get('verdetto', 'NOT ENOUGH INFO'))

        except Exception:
            label_vere.append(riga['label'])
            label_pred.append('NOT ENOUGH INFO')

        barra.progress((i + 1) / totale, text=f'Claim {i+1}/{totale}…')

    barra.empty()

    # Metriche
    try:
        from Sorgente.Valutazione.metriche_ir  import MetricheIR
        from Sorgente.Valutazione.metriche_nlp import MetricheNLP
        import pandas as pd

        rep_ir  = MetricheIR().valuta_sistema(lista_ir, k_valori=[1, 3, 5, 10])
        rep_nlp = MetricheNLP().valuta_sistema(label_vere, label_pred)

        st.success(f'✅ Valutazione completata su {totale} claim.')
        st.markdown('<br>', unsafe_allow_html=True)

        tab_ir, tab_nlp = st.tabs(['📡 IR · Information Retrieval', '🧠 NLP · Classificazione'])

        with tab_ir:
            st.markdown('<br>', unsafe_allow_html=True)
            c1, c2, c3, c4, c5 = st.columns(5)
            for col, k, lbl in [
                (c1, 'MAP',  'MAP'),
                (c2, 'P@1',  'Precision@1'),
                (c3, 'P@5',  'Precision@5'),
                (c4, 'R@5',  'Recall@5'),
                (c5, 'R@10', 'Recall@10'),
            ]:
                with col:
                    mbox(f"{rep_ir.get(k, 0):.3f}", lbl)

            with st.expander("📋 Riepilogo completo metriche IR"):
                st.json({k: round(v, 4) for k, v in rep_ir.items()})

        with tab_nlp:
            st.markdown('<br>', unsafe_allow_html=True)
            d1, d2, d3 = st.columns(3)
            for col, k, lbl in [
                (d1, 'accuracy',    'Accuracy'),
                (d2, 'f1_macro',    'F1 Macro'),
                (d3, 'kappa_cohen', 'Kappa Cohen'),
            ]:
                with col:
                    mbox(f"{rep_nlp.get(k, 0):.3f}", lbl)

            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown('<div class="slabel">Dettaglio per classe</div>', unsafe_allow_html=True)
            per_classe = rep_nlp.get('per_classe', {})
            if per_classe:
                df_pc = pd.DataFrame(per_classe).T
                df_pc.index.name = 'Classe'
                cols_float = df_pc.select_dtypes('float').columns
                st.dataframe(
                    df_pc.style.format({c: '{:.3f}' for c in cols_float}),
                    use_container_width=True
                )

    except Exception as e:
        st.error(f'Errore nel calcolo metriche: {e}')


# ── Pagina Impostazioni ───────────────────────────────────────────

def pagina_impostazioni():
    st.title('🔧 Impostazioni')
    st.caption("Configura i parametri del sistema di fact-checking.")

    st.markdown('<div class="slabel">Recupero documenti</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        cfg.DOCUMENTI_DA_RECUPERARE = st.slider(
            'Documenti Top-K', 1, 10, cfg.DOCUMENTI_DA_RECUPERARE,
            help='Numero di documenti restituiti da BM25 al classificatore NLI'
        )
        sorgente = st.radio(
            'Sorgente corpus',
            ['fever', 'web'],
            format_func=lambda x: '📚 FEVER (locale)' if x == 'fever' else '🌐 Wikipedia Live',
            index=0 if cfg.SORGENTE_CORPUS == 'fever' else 1
        )
        cfg.SORGENTE_CORPUS = sorgente

    with col2:
        if hasattr(cfg, 'RERANKING_ATTIVO'):
            cfg.RERANKING_ATTIVO = st.toggle(
                '🔁 Re-ranking neurale',
                value=cfg.RERANKING_ATTIVO,
                help='Riordina i candidati BM25 con un cross-encoder semantico'
            )
        cfg.ESPANSIONE_ATTIVA = st.toggle(
            '🔤 Espansione WordNet',
            value=cfg.ESPANSIONE_ATTIVA,
            help='Aggiunge sinonimi ai termini della query prima del recupero'
        )

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="slabel">Parametri BM25</div>', unsafe_allow_html=True)
    col3, col4, col5, col6 = st.columns(4)
    with col3:
        st.metric('k1', cfg.K1)
    with col4:
        st.metric('b', cfg.B)
    with col5:
        st.metric('α authority', cfg.PESO_AUTHORITY)
    with col6:
        st.metric('β freshness', cfg.PESO_FRESHNESS)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="slabel">Modello LLM</div>', unsafe_allow_html=True)
    st.info(f"**{cfg.MODELLO_LLM}** · temperatura {cfg.TEMPERATURA} · max {cfg.TOKEN_MASSIMI} token")

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="slabel">Gestione trend</div>', unsafe_allow_html=True)
    trend = carica_trend()
    st.caption(f"Verifiche salvate nel feed: **{len(trend)}** / 12")
    if st.button('🗑️ Svuota feed trend', type='secondary'):
        if os.path.exists(PERCORSO_TREND):
            os.remove(PERCORSO_TREND)
        st.success('Feed svuotato.')
        st.rerun()


# ── Main ──────────────────────────────────────────────────────────

def main():
    init_state()

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        st.error('⚠️ Nessuna API Key trovata. Imposta OPENAI_API_KEY nel file .env')
        return

    try:
        pipeline = carica_pipeline(api_key)
    except FileNotFoundError as e:
        st.error(f'File non trovato: {e}\nEsegui prima i notebook 00 e 02.')
        return

    navbar()

    pagina = st.session_state.pagina
    if pagina == 'home':
        pagina_home(pipeline)
    elif pagina == 'sviluppatori':
        pagina_sviluppatori()
    elif pagina == 'benchmark':
        pagina_benchmark(pipeline)
    elif pagina == 'impostazioni':
        pagina_impostazioni()


if __name__ == '__main__':
    main()