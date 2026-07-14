from datetime import datetime
import wikipediaapi
import requests
from Sorgente.Corpus.corpus_base import CorpusBase
from Sorgente.Preprocessing.preprocesing import Preprocessore
import re
from nltk.corpus import stopwords

TIMESTAMP_OGGI = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')

class CorpusWeb(CorpusBase):

    def __init__(
            self,
            preprocessore: Preprocessore,
            max_pagine: int = 5
    ):
        self._preprocessore = preprocessore
        self._max_pagine = max_pagine
        self._user_agent = 'FactChecker-ProgettoEsame/1.0 (studente@unical.it)'
        self._wiki = wikipediaapi.Wikipedia(
            language='EN',
            user_agent=self._user_agent
        )

    def ottieni_documenti(self, query: str) -> dict:
        corpus_runtime = {}
        titoli_trovati = self._cerca_titoli(query)
        for titolo in titoli_trovati[:self._max_pagine]:
            pagina = self._wiki.page(titolo)
            if not pagina.exists():
                continue
            testo = pagina.text
            lines = self._testo_in_formato_lines(testo)
            id_doc = titolo.replace(' ', '_')
            corpus_runtime[id_doc] = {
                'lines': lines,
                'text': testo,
                'numero_link_entrata': len(testo),
                'timestamp': TIMESTAMP_OGGI
            }
        return corpus_runtime

    def numero_documenti(self) -> int:
        return 0

    def _cerca_titoli(self, query: str) -> list[str]:
        titoli = []
        url = f"https://{self._wiki.language}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "utf8": 1,
            "srlimit": self._max_pagine
        }
        headers = {"User-Agent": self._user_agent}
        try:
            risposta = requests.get(url, params=params, headers=headers)
            dati = risposta.json()
            if "query" in dati and "search" in dati["query"]:
                for risultato in dati["query"]["search"]:
                    titoli.append(risultato["title"])
        except Exception as e:
            print(f"Errore API di ricerca: {e}")
        if not titoli:
            lingua_stop = 'italian' if self._wiki.language == 'it' else 'english'
            stop = set(stopwords.words(lingua_stop))
            parole = re.findall(r'\b[a-zA-Z]{3,}\b', query)
            parole_chiave = [p for p in parole if p.lower() not in stop]
            if parole_chiave:
                query_ridotta = " ".join(parole_chiave[:4])
                params["srsearch"] = query_ridotta
                try:
                    risposta = requests.get(url, params=params, headers=headers)
                    dati = risposta.json()
                    if "query" in dati and "search" in dati["query"]:
                        for risultato in dati["query"]["search"]:
                            titoli.append(risultato["title"])
                except Exception as e:
                    print(f"Errore API di ricerca fallback: {e}")
        return titoli

    def _testo_in_formato_lines(self, testo: str) -> str:
        if not testo:
            return ''
        frasi = [f.strip() for f in testo.split('.') if f.strip()]
        righe = [f'{i}\t{frase}.' for i, frase in enumerate(frasi[:200])]
        return '\n'.join(righe)