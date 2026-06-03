import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus   import stopwords
from nltk.stem     import PorterStemmer

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

PATTERN_ANNOTAZIONI_FEVER = re.compile(r'-LRB-|-RRB-|-LSB-|-RSB-|-LCB-|-RCB-')
PATTERN_NON_ALFANUMERICO = re.compile(r'[^a-z0-9]')


class Preprocessore:

    def __init__(self):
        self.stopwords_inglesi = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def pulisci_annotazioni_fever(self, testo: str) -> str:
        return PATTERN_ANNOTAZIONI_FEVER.sub(' ', testo)

    def tokenizza(self, testo: str) -> list[str]:
        return word_tokenize(testo)

    def converti_minuscolo(self, token: list[str]) -> list[str]:
        return [t.lower() for t in token]

    def rimuovi_punteggiatura(self, token: list[str]) -> list[str]:
        token_puliti = [PATTERN_NON_ALFANUMERICO.sub('', t) for t in token]
        return [t for t in token_puliti if t]

    def rimuovi_stopwords(self, token: list[str]) -> list[str]:
        return [t for t in token if t not in self.stopwords_inglesi]

    def applica_stemming(self, token: list[str]) -> list[str]:
        return [self.stemmer.stem(t) for t in token]

    def elabora(self, testo: str) -> list[str]:
        if not testo or not testo.strip():
            return []
        # Passo 0 — Pulizia annotazioni FEVER
        testo = self.pulisci_annotazioni_fever(testo)
        # Passo 1 — Tokenizzazione
        token = self.tokenizza(testo)
        # Passo 2 — Lowercase
        token = self.converti_minuscolo(token)
        # Passo 3 — Rimozione punteggiatura
        token = self.rimuovi_punteggiatura(token)
        # Passo 4 — Rimozione stopwords
        token = self.rimuovi_stopwords(token)
        # Passo 5 — Stemming
        token = self.applica_stemming(token)
        return token

    def elabora_lista(self, testi: list[str]) -> list[list[str]]:
        return [self.elabora(testo) for testo in testi]