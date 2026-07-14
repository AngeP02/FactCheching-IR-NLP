import os
import pickle
import config as cfg
from Sorgente.Preprocessing.preprocesing import Preprocessore
from Sorgente.Espansione.espansione_query import EspansoreQuery
from Sorgente.Indicizzazione.indice_invertito import IndiceInvertito
from Sorgente.Recupero.metadati import GestoreMetadati
from Sorgente.Recupero.bm25 import CalcolatoreRanking
from Sorgente.Classificazione.classificatore_nli  import ClassificatoreNLI
from Sorgente.Recupero.reranker import Reranker
from Sorgente.Valutazione.metriche_ir import MetricheIR
from Sorgente.Valutazione.metriche_nlp import MetricheNLP


class Pipeline:

    def __init__(self, api_key: str):
        print('Inizializzazione pipeline...')
        self._preprocessore = Preprocessore()
        self._espansore = EspansoreQuery(self._preprocessore)
        print('Preprocessore e espansore query')
        self._corpus = self._carica_corpus()
        self._indice = self._carica_indice()
        print(f'Corpus ({len(self._corpus):,} documenti) e indice '
              f'({self._indice.numero_termini:,} termini)')
        self._metadati = GestoreMetadati(self._corpus)
        self._ranking = CalcolatoreRanking(
            self._indice,
            self._preprocessore,
            self._metadati
        )
        print('Metadati e ranking BM25 esteso')
        self._classificatore = ClassificatoreNLI(api_key)
        self._reranker = Reranker() if cfg.RERANKING_ATTIVO else None
        print('Reranker neurale' if cfg.RERANKING_ATTIVO else '  - Reranker disabilitato')
        print('Classificatore NLI')
        self._metriche_ir = MetricheIR()
        self._metriche_nlp = MetricheNLP()
        print('Moduli metriche')
        print('Pipeline pronta.\n')

    def _carica_corpus(self) -> dict:
        if cfg.SORGENTE_CORPUS == 'fever':
            if not os.path.exists(cfg.PERCORSO_CORPUS_ARRICCHITO):
                raise FileNotFoundError(
                    f"Corpus non trovato: {cfg.PERCORSO_CORPUS_ARRICCHITO}\n"
                    "Esegui prima il notebook 00_studio_dataset.ipynb"
                )
            with open(cfg.PERCORSO_CORPUS_ARRICCHITO, 'rb') as f:
                return pickle.load(f)
        else:
            return {}

    def _carica_indice(self) -> IndiceInvertito:
        indice = IndiceInvertito()
        if cfg.SORGENTE_CORPUS == 'fever':
            if not os.path.exists(cfg.PERCORSO_INDICE):
                raise FileNotFoundError(
                    f"Indice non trovato: {cfg.PERCORSO_INDICE}\n"
                    "Esegui prima il notebook 02_test_indice.ipynb"
                )
            indice.carica(cfg.PERCORSO_INDICE)
        return indice

    def verifica(self, claim: str, sentence_ids_map: dict | None = None) -> dict:
        token_query = self._espansore.espandi(claim)
        if cfg.SORGENTE_CORPUS == 'web':
            risultati_bm25, corpus_da_usare = self._recupera_da_web(claim)
        else:
            n_candidati = cfg.CANDIDATI_RERANKING if cfg.RERANKING_ATTIVO else cfg.DOCUMENTI_DA_RECUPERARE
            risultati_bm25 = self._ranking.recupera_top_k(claim, k=n_candidati, token_query=token_query)
            corpus_da_usare = self._corpus
        if not risultati_bm25:
            return {
                'claim': claim,
                'verdetto': 'NOT ENOUGH INFO',
                'confidenza': 0.0,
                'pesi_label': {},
                'spiegazione': '',
                'documenti': [],
                'token_query': token_query
            }
        if self._reranker and cfg.RERANKING_ATTIVO:
            risultati_bm25 = self._reranker.reordina(
                claim=claim,
                documenti=risultati_bm25,
                corpus=corpus_da_usare,
                top_k=cfg.DOCUMENTI_DA_RECUPERARE
            )
        risultato_nli = self._classificatore.verifica(
            claim=claim,
            risultati_bm25=risultati_bm25,
            corpus_arricchito=corpus_da_usare,
            sentence_ids_map=sentence_ids_map
        )
        return {
            'claim': claim,
            'verdetto': risultato_nli['verdetto'],
            'confidenza': risultato_nli['confidenza'],
            'pesi_label': risultato_nli['pesi_label'],
            'spiegazione': risultato_nli.get('spiegazione', ''),
            'documenti': risultato_nli['dettagli'],
            'token_query': token_query
        }

    def _recupera_da_web(self, claim: str) -> tuple[list[dict], dict]:
        from Sorgente.Corpus.Corpus_web import CorpusWeb
        corpus_web = CorpusWeb(self._preprocessore)
        corpus_runtime = corpus_web.ottieni_documenti(claim)
        if not corpus_runtime:
            return [], {}
        from Sorgente.Indicizzazione.indice_invertito import IndiceInvertito
        indice_temp = IndiceInvertito()
        for id_doc, doc in corpus_runtime.items():
            token = self._preprocessore.elabora(doc.get('text', ''))
            indice_temp.aggiungi_documento(id_doc, token)
        metadati_temp = GestoreMetadati(corpus_runtime)
        ranking_temp = CalcolatoreRanking(
            indice_temp,
            self._preprocessore,
            metadati_temp
        )
        risultati = ranking_temp.recupera_top_k(
            claim, k=cfg.DOCUMENTI_DA_RECUPERARE
        )
        return risultati, corpus_runtime