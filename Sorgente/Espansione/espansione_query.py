import nltk
from nltk.corpus import wordnet
import config as cfg

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class EspansoreQuery:
    def __init__(self, preprocessore, max_sinonimi=3, attivo=cfg.ESPANSIONE_ATTIVA):
        print(f"[EQ] __init__ chiamato con attivo={attivo}")
        self._preprocessore = preprocessore
        self._max_sinonimi = max_sinonimi

    @property
    def _attivo(self):
        return cfg.ESPANSIONE_ATTIVA

    def _e_sinonimo_valido(self, testo_lemma: str) -> bool:
        testo_pulito = testo_lemma.replace('_', ' ')
        prima_parola = testo_pulito.split()[0]
        if prima_parola[0].isupper():
            return False
        return True

    def _espandi_termine(self, termine: str) -> list[str]:
        synsets = wordnet.synsets(termine)
        print(f"\n[DEBUG] termine='{termine}' -> synsets={[s.name() for s in synsets[:3]]}")
        if not synsets:
            print(f"[DEBUG] Nessun synset trovato per '{termine}'")
            return []
        primo_synset = synsets[0]
        stem_originale = self._preprocessore.elabora(termine)
        stem_orig_set = set(stem_originale) if stem_originale else set()
        sinonimi_stem = []
        for lemma in primo_synset.lemmas():
            testo_lemma = lemma.name()
            valido = self._e_sinonimo_valido(testo_lemma)
            testo_pulito = testo_lemma.replace('_', ' ')
            token = self._preprocessore.elabora(testo_pulito)
            stem = token[0] if token else None
            print(f"[DEBUG]   lemma='{testo_lemma}' valido={valido} stem={stem} dup={stem in stem_orig_set or stem in sinonimi_stem}")
            if not valido or not token:
                continue
            if stem in stem_orig_set or stem in sinonimi_stem:
                continue
            sinonimi_stem.append(stem)
            if len(sinonimi_stem) >= self._max_sinonimi:
                break
        print(f"[DEBUG] sinonimi trovati per '{termine}': {sinonimi_stem}")
        return sinonimi_stem

    def espandi(self, query: str) -> list[str]:
        print(f"\n[EQ] espandi() chiamato con query='{query}'")
        print(f"[EQ] cfg.ESPANSIONE_ATTIVA = {cfg.ESPANSIONE_ATTIVA}")  # ← aggiungi questa
        print(f"[EQ] self._attivo = {self._attivo}")
        token_originali = self._preprocessore.elabora(query)
        print(f"[EQ] token_originali (stem): {token_originali}")
        if not token_originali:
            return []
        if not self._attivo:
            print(f"[EQ] ✗ espansione DISATTIVA, ritorno stem")
            return token_originali
        token_grezzi = self._preprocessore.elabora_senza_stemming(query)
        print(f"[EQ] token_grezzi (no stem): {token_grezzi}")
        return self.espandi_token(token_originali, token_grezzi)

    def espandi_token(self, token_originali: list[str], token_grezzi: list[str] | None = None) -> list[str]:
        print(f"\n[EQ] espandi_token() chiamato")
        print(f"[EQ]   token_originali: {token_originali}")
        print(f"[EQ]   token_grezzi:    {token_grezzi}")
        if not self._attivo:
            print(f"[EQ] ✗ espansione DISATTIVA")
            return list(token_originali)
        sorgente_espansione = token_grezzi if token_grezzi is not None else token_originali
        print(f"[EQ]   sorgente_espansione: {sorgente_espansione}")
        token_espansi = list(token_originali)
        insieme_token = set(token_originali)
        for token_grezzo in sorgente_espansione:
            sinonimi = self._espandi_termine(token_grezzo)
            print(f"[EQ]   '{token_grezzo}' → sinonimi: {sinonimi}")
            for sinonimo in sinonimi:
                if sinonimo not in insieme_token:
                    token_espansi.append(sinonimo)
                    insieme_token.add(sinonimo)
        print(f"[EQ] token_espansi finali: {token_espansi}")
        return token_espansi

    def statistiche_espansione(self, query: str) -> dict:
        token_orig = self._preprocessore.elabora(query)
        token_grezzi = self._preprocessore.elabora_senza_stemming(query)
        token_esp = self.espandi_token(token_orig, token_grezzi)
        aggiunti = [t for t in token_esp if t not in set(token_orig)]
        return {
            'token_originali': token_orig,
            'token_espansi': token_esp,
            'sinonimi_aggiunti': aggiunti,
            'n_originali': len(token_orig),
            'n_aggiunti': len(aggiunti)
        }