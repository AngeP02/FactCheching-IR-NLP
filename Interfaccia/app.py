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

st.set_page_config(
    page_title='Fact Checker AI',
    page_icon='🔍',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# ── Percorso file trend ───────────────────────────────────────────
PERCORSO_TREND = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'trend_verifiche.json'
)

# ── Colori verdetto ───────────────────────────────────────────────
COLORI_VERDETTO = {
    'SUPPORTS':       '🟢',
    'REFUTES':        '🔴',
    'NOT ENOUGH INFO':'🟡',
}

# ── Gestione trend ────────────────────────────────────────────────

def carica_trend() -> list[dict]:
    if not os.path.exists(PERCORSO_TREND):
        return []
    try:
        with open(PERCORSO_TREND, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []


def salva_trend(claim: str, verdetto: str, confidenza: float):
    trend = carica_trend()
    nuova = {
        'claim':      claim,
        'verdetto':   verdetto,
        'confidenza': confidenza,
        'data':       datetime.datetime.now().strftime('%d/%m/%Y %H:%M')
    }
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

def etichetta_verdetto(v: str) -> str:
    return f"{icona(v)} {v}"

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
    pagine = {
        'home':         'Home',
        'sviluppatori': 'Analisi Tecnica',
        'benchmark':    'Benchmark',
        'impostazioni': 'Impostazioni',
    }
    cols = st.columns([3, 1, 1, 1, 1])
    with cols[0]:
        st.subheader('🔍 FactCheck AI')
    for col, (chiave, etichetta) in zip(cols[1:], pagine.items()):
        with col:
            tipo = 'primary' if st.session_state.pagina == chiave else 'secondary'
            if st.button(etichetta, key=f'nav_{chiave}', type=tipo, use_container_width=True):
                st.session_state.pagina = chiave
                st.rerun()
    st.divider()


# ── Pagina Home ───────────────────────────────────────────────────

def pagina_home(pipeline: Pipeline):
    st.caption("Impostazioni base di configurazione")

    # Il container con bordo racchiude tutto in un riquadro pulito
    with st.container(border=True):

        # vertical_alignment="center" (o "bottom") allinea automaticamente i toggle all'input numerico
        c1, c2, c3, c4, c5 = st.columns(5, vertical_alignment="center")

        with c1:
            cfg.DOCUMENTI_DA_RECUPERARE = st.number_input(
                label="Numero di documenti Top-K da recuperare:",
                min_value=1,
                max_value=10,
                value=int(cfg.DOCUMENTI_DA_RECUPERARE) if hasattr(cfg, 'DOCUMENTI_DA_RECUPERARE') else 5,
                step=1,
                format="%d"
            )

        with c3:
            st.write("")
            st.write("")
            if hasattr(cfg, 'RERANKING_ATTIVO'):
                cfg.RERANKING_ATTIVO = st.toggle(
                    'Re-ranking neurale',
                    value=cfg.RERANKING_ATTIVO,
                )

        with c5:
            st.write("")
            st.write("")
            if hasattr(cfg, 'ESPANSIONE_ATTIVA'):
                cfg.ESPANSIONE_ATTIVA = st.toggle(
                    "Espansione WordNet",
                    value=cfg.ESPANSIONE_ATTIVA,
                )

    st.write("")
    st.caption("Inserisci un'affermazione in inglese: il sistema cercherà le evidenze e ti darà un verdetto motivato.")

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        claim = st.text_input(
            label='Affermazione da verificare',
            label_visibility='collapsed',
            placeholder='Es: The Eiffel Tower is located in Paris, France.',
            value=st.session_state.last_claim,
            key='input_claim'
        )
    with col_btn:
        verifica = st.button('Verifica', type='primary', use_container_width=True)

    # Verifica
    if verifica and claim.strip():
        st.session_state.last_claim = claim
        with st.spinner('Recupero evidenze e classificazione in corso'):
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

    # Risultato
    risultato = st.session_state.last_result
    if risultato and risultato['claim'] == st.session_state.last_claim:
        _mostra_risultato(risultato, st.session_state.last_stats)

    st.write('')
    _mostra_trend_feed()


def _mostra_risultato(risultato: dict, stats):
    verdetto = risultato['verdetto']
    confidenza = risultato['confidenza']
    spiegazione = risultato.get('spiegazione', '')
    documenti = risultato.get('documenti', [])
    pesi = risultato.get('pesi_label', {})
    token = risultato.get('token_query', [])

    st.divider()
    col_sx, col_dx = st.columns([3, 2], gap='large')

    with (col_sx):
        with st.container(border=True):
            st.caption('Claim verificata')
            st.markdown(f"#### *{risultato['claim']}*")

            # Verdetto con stile compatto (usa un container colorato come badge)
            st.caption("Verdetto finale")
            verdetto_label = etichetta_verdetto(verdetto)
            # Colore basato sul verdetto (es. verde per SUPPORTS, rosso per REFUTES)
            colore = "green" if verdetto == "SUPPORTS" else "red" if verdetto == "REFUTES" else "gray"
            if verdetto == "SUPPORTS":
                st.success(verdetto_label)
            elif verdetto == "REFUTES":
                st.error(verdetto_label)
            else:
                st.warning(verdetto_label)

            # Spiegazione
            if spiegazione:
                st.write("")
                st.caption("Spiegazione della risposta")
                st.write(spiegazione)



    with col_dx:
        with st.container(border=True):
            if pesi:
                st.caption('Distribuzione confidenza')
                totale = max(sum(pesi.values()), 0.001)
                for label in ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']:
                    val = pesi.get(label, 0)
                    pct = val / totale
                    st.progress(pct, text=f"{label} ({val:.1f})")
            st.write("")
            st.write("")
            st.caption("Valore confidenza del sistema")
            st.header(f'{confidenza:.0f}%')
            st.write('')

            if stats and stats.get('n_aggiunti', 0) > 0:
                st.write('')
                st.caption('ESPANSIONE WORDNET')
                col_a, col_b = st.columns(2)
                with col_a:
                    st.caption('Originali')
                    st.code('  '.join(stats.get('token_originali', [])), language=None)
                with col_b:
                    st.caption('Aggiunti')
                    st.code('  '.join(stats.get('sinonimi_aggiunti', [])), language=None)

    # Documenti
    if documenti:
        st.write('')
        with st.expander(f'Dettaglio {len(documenti)} fonti analizzate'):
            for i, doc in enumerate(documenti, 1):
                _render_doc(doc, i)


def _render_doc(doc: dict, rank: int):
    titolo = doc.get('titolo', doc.get('id_doc', ''))
    verdetto_doc = doc.get('verdetto', '')
    ev = doc.get('evidenza', '')

    with st.container(border=True):
        col_t, col_v = st.columns([4, 1])
        with col_t:
            st.write(f'**{rank}. {titolo}**')
        with col_v:
            st.write(etichetta_verdetto(verdetto_doc))

        meta = f"BM25: `{doc.get('punteggio', 0):.4f}`"
        if 'score_reranking' in doc:
            meta += f"  ·  Re-rank: `{doc['score_reranking']:.4f}`"
        st.caption(meta)

        if ev:
            st.caption(ev[:350] + ('…' if len(ev) > 350 else ''))


def _mostra_trend_feed():
    trend = carica_trend()
    if not trend:
        return

    st.caption('ULTIME VERIFICHE')
    righe = [trend[i:i+3] for i in range(0, min(len(trend), 9), 3)]
    for riga in righe:
        cols = st.columns(3, gap='medium')
        for col, voce in zip(cols, riga):
            with col:
                with st.container(border=True):
                    st.caption(voce['data'])
                    st.write(f"**{voce['claim'][:100]}{'…' if len(voce['claim']) > 100 else ''}**")
                    st.write(etichetta_verdetto(voce['verdetto']))
                    st.caption(f"Confidenza: {voce['confidenza']:.0f}%")


# ── Pagina Analisi Tecnica ────────────────────────────────────────

def pagina_sviluppatori():
    st.title('⚙️ Analisi Tecnica')
    st.caption("Dettagli IR e NLP dell'ultima query elaborata.")

    risultato = st.session_state.last_result
    if not risultato:
        st.info('Esegui prima una verifica dalla Home.')
        return

    verdetto    = risultato['verdetto']
    confidenza  = risultato['confidenza']
    pesi        = risultato.get('pesi_label', {})
    spiegazione = risultato.get('spiegazione', '')
    documenti   = risultato.get('documenti', [])
    token       = risultato.get('token_query', [])
    stats       = st.session_state.last_stats

    if token:
        st.caption('Token query')
        st.code('  '.join(token), language=None)

    st.write(f"*\"{risultato['claim'][:120]}{'…' if len(risultato['claim']) > 120 else ''}\"*")
    col_v, col_c, _ = st.columns([1, 1, 2])
    with col_v:
        st.metric('Verdetto', etichetta_verdetto(verdetto))
    with col_c:
        st.metric('Confidenza', f'{confidenza:.1f}%')
    st.divider()

    tab_nlp, tab_ir = st.tabs(['🧠 NLP · Classificazione', '📄 IR · Documenti'])

    with tab_nlp:
        col_nlp, col_conf = st.columns([2, 1], gap='large')

        with col_nlp:
            if pesi:
                st.caption('PESI PER LABEL')
                totale = max(sum(pesi.values()), 0.001)
                for label in ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']:
                    val = pesi.get(label, 0)
                    st.progress(val / totale, text=f'{icona(label)}  {label}  —  {val:.2f}')

            if spiegazione:
                st.write('')
                st.caption('SPIEGAZIONE LLM')
                st.info(spiegazione)

            if token:
                st.write('')
                st.caption('TOKEN QUERY')
                st.code('  '.join(token), language=None)

            if stats and stats.get('n_aggiunti', 0) > 0:
                st.write('')
                st.caption('ESPANSIONE WORDNET')
                col_a, col_b = st.columns(2)
                with col_a:
                    st.caption('Originali')
                    st.code('  '.join(stats.get('token_originali', [])), language=None)
                with col_b:
                    st.caption('Aggiunti')
                    st.code('  '.join(stats.get('sinonimi_aggiunti', [])), language=None)

        with col_conf:
            st.metric('Confidenza', f'{confidenza:.0f}%')
            st.progress(int(confidenza) / 100)

    with tab_ir:
        if documenti:
            for i, doc in enumerate(documenti, 1):
                _render_doc(doc, i)
        else:
            st.info('Nessun documento recuperato.')


# ── Pagina Benchmark ──────────────────────────────────────────────

def pagina_benchmark(pipeline: Pipeline):
    st.title('📊 Benchmark')
    st.caption('Valuta il sistema su un campione del dataset FEVER.')

    st.info(
        'Il benchmark carica automaticamente il dataset FEVER da HuggingFace '
        '(già in cache se hai eseguito i notebook). Non è necessario nessun file esterno.'
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

    with st.spinner('Caricamento dataset FEVER…'):
        try:
            from datasets import load_dataset
            import pandas as pd

            ds     = load_dataset('fever', 'v1.0')
            df_raw = pd.DataFrame(ds[split_scelto])
            df     = df_raw.groupby('id').agg(
                claim         =('claim',                'first'),
                label         =('label',                'first'),
                url_documenti =('evidence_wiki_url',    lambda x: list(x.dropna().unique())),
                sentence_ids  =('evidence_sentence_id', list)
            ).reset_index()

            campioni = []
            for lbl in ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']:
                sub = df[df['label'] == lbl]
                campioni.append(sub.sample(n=min(n_per_classe, len(sub)), random_state=42))
            df_campione = pd.concat(campioni).reset_index(drop=True)
            st.success(f'Dataset caricato: {len(df_campione)} claim ({n_per_classe} per classe)')

        except Exception as e:
            st.error(f'Errore nel caricamento del dataset: {e}')
            return

    lista_ir   = []
    label_vere = []
    label_pred = []
    totale     = len(df_campione)
    barra      = st.progress(0, text='Valutazione claim…')

    for i, (_, riga) in enumerate(df_campione.iterrows()):
        try:
            sids_map = {}
            for url, sid in zip(riga['url_documenti'], riga['sentence_ids']):
                if url:
                    sids_map.setdefault(url, []).append(sid)

            ris       = pipeline.verifica(riga['claim'], sentence_ids_map=sids_map or None)
            docs      = [d['id_doc'] for d in ris.get('documenti', [])]
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

    try:
        from Sorgente.Valutazione.metriche_ir  import MetricheIR
        from Sorgente.Valutazione.metriche_nlp import MetricheNLP
        import pandas as pd

        rep_ir  = MetricheIR().valuta_sistema(lista_ir, k_valori=[1, 3, 5, 10])
        rep_nlp = MetricheNLP().valuta_sistema(label_vere, label_pred)

        st.success(f'✅ Valutazione completata su {totale} claim.')

        tab_ir, tab_nlp = st.tabs(['📡 IR · Information Retrieval', '🧠 NLP · Classificazione'])

        with tab_ir:
            st.write('')
            c1, c2, c3, c4, c5 = st.columns(5)
            for col, k, lbl in [
                (c1, 'MAP',  'MAP'),
                (c2, 'P@1',  'Precision@1'),
                (c3, 'P@5',  'Precision@5'),
                (c4, 'R@5',  'Recall@5'),
                (c5, 'R@10', 'Recall@10'),
            ]:
                with col:
                    st.metric(lbl, f"{rep_ir.get(k, 0):.3f}")

            with st.expander('📋 Riepilogo completo metriche IR'):
                st.json({k: round(v, 4) for k, v in rep_ir.items()})

        with tab_nlp:
            st.write('')
            d1, d2, d3 = st.columns(3)
            for col, k, lbl in [
                (d1, 'accuracy',    'Accuracy'),
                (d2, 'f1_macro',    'F1 Macro'),
                (d3, 'kappa_cohen', 'Kappa Cohen'),
            ]:
                with col:
                    st.metric(lbl, f"{rep_nlp.get(k, 0):.3f}")

            st.write('')
            st.caption('DETTAGLIO PER CLASSE')
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
    st.caption('Configura i parametri del sistema di fact-checking.')

    st.subheader('Recupero documenti')
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
                '🔁 Re-ranking neurale', value=cfg.RERANKING_ATTIVO,
                help='Riordina i candidati BM25 con un cross-encoder semantico'
            )
        cfg.ESPANSIONE_ATTIVA = st.toggle(
            '🔤 Espansione WordNet', value=cfg.ESPANSIONE_ATTIVA,
            help='Aggiunge sinonimi ai termini della query prima del recupero'
        )

    st.divider()
    st.subheader('Parametri BM25')
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric('k1', cfg.K1)
    with c2: st.metric('b', cfg.B)
    with c3: st.metric('α authority', cfg.PESO_AUTHORITY)
    with c4: st.metric('β freshness', cfg.PESO_FRESHNESS)

    st.divider()
    st.subheader('Modello LLM')
    st.info(f"**{cfg.MODELLO_LLM}** · temperatura {cfg.TEMPERATURA} · max {cfg.TOKEN_MASSIMI} token")

    st.divider()
    st.subheader('Gestione trend')
    trend = carica_trend()
    st.caption(f"Verifiche salvate nel feed: **{len(trend)}** / 12")
    if st.button('🗑️ Svuota feed trend'):
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