class MetricheIR:

    def precision_at_k(self, recuperati: list[str], rilevanti: set[str], k: int) -> float:
        if not recuperati or k == 0:
            return 0.0
        top_k = recuperati[:k]
        hit = sum(1 for doc in top_k if doc in rilevanti)
        return hit / k

    def recall_at_k(self, recuperati: list[str], rilevanti: set[str], k: int) -> float:
        if not rilevanti:
            return 0.0
        top_k = set(recuperati[:k])
        hit = len(top_k & rilevanti)
        return hit / len(rilevanti)

    def f1_at_k(self, recuperati: list[str], rilevanti: set[str], k: int) -> float:
        p = self.precision_at_k(recuperati, rilevanti, k)
        r = self.recall_at_k(recuperati, rilevanti, k)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    def average_precision(self, recuperati: list[str], rilevanti: set[str]) -> float:
        if not rilevanti or not recuperati:
            return 0.0
        somma_precision = 0.0
        hit_cumulativi  = 0
        for posizione, id_doc in enumerate(recuperati, start=1):
            if id_doc in rilevanti:
                hit_cumulativi += 1
                somma_precision += hit_cumulativi / posizione
        return somma_precision / len(rilevanti)

    def mean_average_precision(self, lista_risultati: list[dict]) -> float:
        if not lista_risultati:
            return 0.0
        somma_ap = sum(
            self.average_precision(r['recuperati'], r['rilevanti'])
            for r in lista_risultati
        )
        return somma_ap / len(lista_risultati)

    def valuta_sistema(self, lista_risultati: list[dict], k_valori: list[int] = None) -> dict:
        if k_valori is None:
            k_valori = [1, 3, 5, 10]
        n = len(lista_risultati)
        if n == 0:
            return {}
        risultato = {'MAP': self.mean_average_precision(lista_risultati)}
        for k in k_valori:
            somma_p = sum(
                self.precision_at_k(r['recuperati'], r['rilevanti'], k)
                for r in lista_risultati
            )
            somma_r = sum(
                self.recall_at_k(r['recuperati'], r['rilevanti'], k)
                for r in lista_risultati
            )
            somma_f1 = sum(
                self.f1_at_k(r['recuperati'], r['rilevanti'], k)
                for r in lista_risultati
            )
            risultato[f'P@{k}'] = round(somma_p  / n, 4)
            risultato[f'R@{k}'] = round(somma_r  / n, 4)
            risultato[f'F1@{k}'] = round(somma_f1 / n, 4)
        risultato['MAP'] = round(risultato['MAP'], 4)
        return risultato