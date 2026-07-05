import nltk
from nltk.corpus import wordnet

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from Sorgente.Preprocessing.preprocesing import Preprocessore
import config as cfg


class EspansoreQuery:
    def __init__(self, preprocessore: Preprocessore, max_sinonimi:  int  = 3, attivo: bool = cfg.ESPANSIONE_ATTIVA):
        self._preprocessore = preprocessore
        self._max_sinonimi  = max_sinonimi
        self._attivo        = attivo

    def _e_sinonimo_valido(self, testo_lemma: str) -> bool:
        testo_pulito = testo_lemma.replace('_', ' ')
        prima_parola = testo_pulito.split()[0]
        if prima_parola[0].isupper():
            return False
        return True

    def _espandi_termine(self, termine: str) -> list[str]:
        synsets = wordnet.synsets(termine)
        if not synsets:
            return []
        primo_synset = synsets[0]
        stem_originale = self._preprocessore.elabora(termine)
        stem_orig_set = set(stem_originale) if stem_originale else set()
        sinonimi_stem = []
        for lemma in primo_synset.lemmas():
            testo_lemma = lemma.name()
            if not self._e_sinonimo_valido(testo_lemma):
                continue
            testo_pulito = testo_lemma.replace('_', ' ')
            token = self._preprocessore.elabora(testo_pulito)
            if not token:
                continue
            stem = token[0]
            if stem in stem_orig_set or stem in sinonimi_stem:
                continue
            sinonimi_stem.append(stem)
            if len(sinonimi_stem) >= self._max_sinonimi:
                break
        return sinonimi_stem

    def espandi(self, query: str) -> list[str]:
        token_originali = self._preprocessore.elabora(query)
        if not token_originali:
            return []
        if not self._attivo:
            return token_originali
        return self.espandi_token(token_originali)

    def espandi_token(self, token_originali: list[str]) -> list[str]:
        if not self._attivo:
            return list(token_originali)
        token_espansi = list(token_originali)
        insieme_token = set(token_originali)
        for token in token_originali:
            sinonimi = self._espandi_termine(token)
            for sinonimo in sinonimi:
                if sinonimo not in insieme_token:
                    token_espansi.append(sinonimo)
                    insieme_token.add(sinonimo)
        return token_espansi

    def statistiche_espansione(self, query: str) -> dict:
        token_orig = self._preprocessore.elabora(query)
        token_esp = self.espandi(query)
        aggiunti = [t for t in token_esp if t not in set(token_orig)]
        return {
            'token_originali': token_orig,
            'token_espansi': token_esp,
            'sinonimi_aggiunti': aggiunti,
            'n_originali': len(token_orig),
            'n_aggiunti': len(aggiunti)
        }