_PROMPT_SISTEMA = """You are a fact-checking assistant with NO prior knowledge.
Your task is to determine whether a given EVIDENCE supports, refutes, or is
insufficient to verify a CLAIM.

You must respond with EXACTLY one of these three labels:
  SUPPORTS         - the evidence EXPLICITLY contains information that supports the claim
  REFUTES          - the evidence EXPLICITLY contains information that contradicts the claim
  NOT ENOUGH INFO  - the evidence does not explicitly mention the information needed

Rules:
  - Respond with the label only. No explanation, no punctuation.
  - Base your judgment ONLY on the exact words in the provided evidence.
  - If the evidence does not EXPLICITLY state the fact in the claim, respond NOT ENOUGH INFO.
  - NEVER use your training knowledge. Treat yourself as if you have never heard of any person, place or event before.
  - If you are unsure, respond NOT ENOUGH INFO."""

class CostruttorePrompt:
    def __init__(self):
        self._prompt_sistema = _PROMPT_SISTEMA

    @property
    def prompt_sistema(self) -> str:
        return self._prompt_sistema

    def costruisci_prompt_utente(self, claim: str, evidenza: str) -> str:
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
        return self._prompt_sistema, self.costruisci_prompt_utente(claim, evidenza)

    def prepara_messaggi(
        self,
        claim:    str,
        evidenza: str
    ) -> list[dict]:

        return [
            {'role': 'system', 'content': self._prompt_sistema},
            {'role': 'user',   'content': self.costruisci_prompt_utente(claim, evidenza)}
        ]