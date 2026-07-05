"""
costruttore_prompt.py
════════════════════════════════════════════════════════════════════
Costruzione dei prompt per la classificazione NLI tramite LLM.

Responsabilità:
    Preparare i messaggi da inviare all'LLM per classificare la
    relazione tra una claim e un'evidenza testuale.

Perché un modulo separato:
    Il prompt è la componente più sensibile del sistema NLI —
    piccole variazioni nel testo producono risposte diverse.
    Separarlo in un modulo dedicato permette di modificarlo,
    testarlo e confrontare varianti senza toccare la logica
    di classificazione.

Struttura del prompt:
    Il prompt è diviso in due parti:
    1. Sistema: definisce il ruolo e le regole permanenti
    2. Utente:  fornisce l'evidenza e la claim per ogni richiesta

    La struttura "EVIDENCE / CLAIM / Response:" è una tecnica
    di prompt engineering che:
    - Separa chiaramente le due componenti semantiche
    - Forza l'LLM a completare con la label (non con testo libero)
    - Riduce la probabilità di output non strutturati

Limiti noti:
    L'LLM può usare conoscenza esterna nonostante l'istruzione
    "Do not use external knowledge". Questo è un limite intrinseco
    dei modelli linguistici pre-addestrati — non completamente
    eliminabile tramite prompt engineering.
    Esempio osservato: "Obama was born in 1985" → REFUTES invece
    di NOT ENOUGH INFO, perché il modello sa che un presidente
    non può essere nato nel 1985.

Utilizzo:
    from sorgente.classificazione.costruttore_prompt import CostruttorePrompt

    costruttore = CostruttorePrompt()
    sistema, utente = costruttore.costruisci('Obama was born in Hawaii',
                                             'Barack Obama is a politician...')
════════════════════════════════════════════════════════════════════
"""


# Prompt di sistema — istruzioni permanenti per l'LLM.
# Definisce ruolo, compito, label valide e regole di output.
# Scritto in inglese perché il corpus FEVER è in inglese e
# i modelli LLM performano meglio nella lingua del corpus.
_PROMPT_SISTEMA = """You are a fact-checking assistant.
Your task is to determine whether a given EVIDENCE supports, refutes, or is
insufficient to verify a CLAIM.

You must respond with EXACTLY one of these three labels:
  SUPPORTS         - the evidence clearly supports the claim
  REFUTES          - the evidence clearly contradicts the claim
  NOT ENOUGH INFO  - the evidence is insufficient to verify the claim

Rules:
  - Respond with the label only. No explanation, no punctuation.
  - Base your judgment solely on the provided evidence.
  - Do not use external knowledge."""


class CostruttorePrompt:
    """
    Costruisce i prompt per la classificazione NLI.

    Fornisce il prompt di sistema (fisso) e il prompt utente
    (variabile per ogni coppia claim-evidenza).

    Il prompt di sistema viene caricato una volta sola e
    riusato per tutte le richieste — nessun overhead ripetuto.
    """

    def __init__(self):
        """
        Inizializza il costruttore caricando il prompt di sistema.
        """
        self._prompt_sistema = _PROMPT_SISTEMA

    # ── Interfaccia pubblica ──────────────────────────────────────

    @property
    def prompt_sistema(self) -> str:
        """
        Restituisce il prompt di sistema fisso.

        Va inviato come messaggio 'system' all'API OpenAI.
        È lo stesso per tutte le richieste di classificazione.

        Returns:
            Stringa del prompt di sistema
        """
        return self._prompt_sistema

    def costruisci_prompt_utente(self, claim: str, evidenza: str) -> str:
        """
        Costruisce il messaggio utente per una specifica coppia claim-evidenza.

        Struttura:
            EVIDENCE: <testo dell'evidenza>

            CLAIM: <testo della claim>

            Response:

        Il tag 'Response:' finale è intenzionale: forza l'LLM a
        completare il testo con la label senza aggiungere prefazioni
        o spiegazioni. È una tecnica standard di prompt engineering
        per ottenere output strutturati da modelli generativi.

        Args:
            claim:    testo della claim da verificare
            evidenza: testo del documento usato come evidenza

        Returns:
            Stringa del prompt utente
        """
        return (
            f"EVIDENCE: {evidenza}\n\n"
            f"CLAIM: {claim}\n\n"
            f"Response:"
        )

    def costruisci(
        self,
        claim:    str,
        evidenza: str
    ) -> tuple[str, str]:
        """
        Restituisce la coppia (prompt_sistema, prompt_utente).

        Metodo di convenienza per ottenere entrambi i messaggi
        in un'unica chiamata, pronti da passare all'API.

        Args:
            claim:    testo della claim
            evidenza: testo dell'evidenza

        Returns:
            Tupla (prompt_sistema, prompt_utente)
        """
        return self._prompt_sistema, self.costruisci_prompt_utente(claim, evidenza)

    def prepara_messaggi(
        self,
        claim:    str,
        evidenza: str
    ) -> list[dict]:
        """
        Prepara la lista di messaggi nel formato OpenAI API.

        Restituisce la struttura pronta da passare direttamente
        al parametro `messages` di `client.chat.completions.create()`.

        Args:
            claim:    testo della claim
            evidenza: testo dell'evidenza

        Returns:
            Lista di dizionari nel formato:
            [
                {'role': 'system', 'content': <prompt_sistema>},
                {'role': 'user',   'content': <prompt_utente>}
            ]
        """
        return [
            {'role': 'system', 'content': self._prompt_sistema},
            {'role': 'user',   'content': self.costruisci_prompt_utente(claim, evidenza)}
        ]