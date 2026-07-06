"""
metriche_ir.py
════════════════════════════════════════════════════════════════════
Metriche di valutazione per il modulo di Information Retrieval.

Responsabilità:
    Calcolare le metriche standard di IR per valutare la qualità
    del ranking BM25 esteso rispetto al ground truth FEVER.

Metriche implementate:

    Precision@K (P@K):
        Frazione di documenti rilevanti tra i primi K recuperati.
        P@K = |{rilevanti} ∩ {top-K recuperati}| / K
        Misura la precisione del sistema nei top-K risultati.
        Valori alti → pochi risultati irrilevanti tra i primi K.

    Recall@K (R@K):
        Frazione di documenti rilevanti trovati tra i top-K.
        R@K = |{rilevanti} ∩ {top-K recuperati}| / |{rilevanti}|
        Misura quanti documenti rilevanti il sistema riesce a trovare.
        Valori alti → il sistema non perde documenti importanti.

    Average Precision (AP):
        Media delle precision calcolate ad ogni posizione in cui
        appare un documento rilevante nella lista ordinata.
        Premia i sistemi che mettono i documenti rilevanti in cima.

    Mean Average Precision (MAP):
        Media degli AP su tutte le query del test set.
        È la metrica principale per confrontare sistemi di IR.
        Considera sia precision che ordinamento dei risultati.

    F1@K:
        Media armonica di P@K e R@K.
        F1@K = 2 * P@K * R@K / (P@K + R@K)
        Utile quando precisione e recall hanno uguale importanza.

Nota sul ground truth FEVER:
    Il dataset FEVER indica per ogni claim quali documenti Wikipedia
    sono rilevanti (evidence_wiki_url). Usiamo questi come ground truth
    per valutare se il sistema li recupera correttamente.
    Le claim NOT ENOUGH INFO non hanno documenti rilevanti e vengono
    escluse dalla valutazione IR (non ha senso misurare il recupero
    se non esiste un documento corretto da trovare).

Utilizzo:
    from sorgente.valutazione.metriche_ir import MetricheIR

    metriche = MetricheIR()
    p5  = metriche.precision_at_k(recuperati, rilevanti, k=5)
    map_score = metriche.mean_average_precision(lista_risultati)
════════════════════════════════════════════════════════════════════
"""


class MetricheIR:
    """
    Calcola le metriche di valutazione per il sistema di IR.

    Tutte le metriche operano su insiemi di ID documento:
    - recuperati: lista ordinata di id_doc restituiti dal sistema
    - rilevanti:  insieme di id_doc corretti (ground truth FEVER)
    """

    # ── Metriche per singola query ────────────────────────────────

    def precision_at_k(
        self,
        recuperati: list[str],
        rilevanti:  set[str],
        k:          int
    ) -> float:
        """
        Calcola la Precision@K per una singola query.

        Misura quanti dei primi K documenti recuperati sono rilevanti.
        P@K = |{rilevanti} ∩ {top-K}| / K

        Esempio:
            recuperati = [A, B, C, D, E]  (A, C rilevanti)
            rilevanti  = {A, C, F}
            P@3 = |{A,C}| / 3 = 2/3 = 0.667

        Args:
            recuperati: lista ordinata di id_doc (dal più al meno rilevante)
            rilevanti:  insieme di id_doc ground truth
            k:          numero di posizioni da considerare

        Returns:
            Valore P@K in [0, 1]. 0.0 se recuperati è vuoto o k=0.
        """
        if not recuperati or k == 0:
            return 0.0

        top_k    = recuperati[:k]
        hit      = sum(1 for doc in top_k if doc in rilevanti)
        return hit / k

    def recall_at_k(
        self,
        recuperati: list[str],
        rilevanti:  set[str],
        k:          int
    ) -> float:
        """
        Calcola il Recall@K per una singola query.

        Misura quanti dei documenti rilevanti compaiono nei top-K.
        R@K = |{rilevanti} ∩ {top-K}| / |{rilevanti}|

        Esempio:
            recuperati = [A, B, C, D, E]  (A, C rilevanti)
            rilevanti  = {A, C, F}
            R@3 = |{A,C}| / |{A,C,F}| = 2/3 = 0.667

        Args:
            recuperati: lista ordinata di id_doc
            rilevanti:  insieme di id_doc ground truth
            k:          numero di posizioni da considerare

        Returns:
            Valore R@K in [0, 1]. 0.0 se rilevanti è vuoto.
        """
        if not rilevanti:
            return 0.0

        top_k = set(recuperati[:k])
        hit   = len(top_k & rilevanti)
        return hit / len(rilevanti)

    def f1_at_k(
        self,
        recuperati: list[str],
        rilevanti:  set[str],
        k:          int
    ) -> float:
        """
        Calcola F1@K — media armonica di P@K e R@K.

        F1@K = 2 * P@K * R@K / (P@K + R@K)

        Utile come metrica singola che bilancia precisione e recall.

        Args:
            recuperati: lista ordinata di id_doc
            rilevanti:  insieme di id_doc ground truth
            k:          numero di posizioni da considerare

        Returns:
            Valore F1@K in [0, 1]. 0.0 se P@K + R@K = 0.
        """
        p = self.precision_at_k(recuperati, rilevanti, k)
        r = self.recall_at_k(recuperati, rilevanti, k)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    def average_precision(
        self,
        recuperati: list[str],
        rilevanti:  set[str]
    ) -> float:
        """
        Calcola l'Average Precision (AP) per una singola query.

        AP è la media delle precision alle posizioni in cui
        compare un documento rilevante nella lista ordinata.

        AP = (1/|R|) * Σ P@k * rel(k)
        dove rel(k)=1 se il documento in posizione k è rilevante.

        Premia i sistemi che posizionano i documenti rilevanti
        in cima alla lista — due sistemi con lo stesso R@10
        avranno AP diversi se uno mette i rilevanti in posizione
        1-2 e l'altro in posizione 8-10.

        Esempio:
            recuperati = [A, B, C, D, E]  (A in pos.1, C in pos.3)
            rilevanti  = {A, C}
            AP = (P@1 + P@3) / 2 = (1/1 + 2/3) / 2 = 0.833

        Args:
            recuperati: lista ordinata di id_doc
            rilevanti:  insieme di id_doc ground truth

        Returns:
            Valore AP in [0, 1]. 0.0 se rilevanti è vuoto.
        """
        if not rilevanti or not recuperati:
            return 0.0

        somma_precision = 0.0
        hit_cumulativi  = 0

        for posizione, id_doc in enumerate(recuperati, start=1):
            if id_doc in rilevanti:
                hit_cumulativi  += 1
                somma_precision += hit_cumulativi / posizione

        return somma_precision / len(rilevanti)

    # ── Metriche aggregate su più query ──────────────────────────

    def mean_average_precision(
        self,
        lista_risultati: list[dict]
    ) -> float:
        """
        Calcola il Mean Average Precision (MAP) su un insieme di query.

        MAP = (1/|Q|) * Σ AP(q)  per ogni query q in Q

        È la metrica principale per confrontare sistemi di IR.
        Considera sia la precisione che l'ordinamento dei risultati
        su tutte le query del test set.

        Args:
            lista_risultati: lista di dizionari, uno per query:
                {
                    'recuperati': list[str],  # id_doc ordinati
                    'rilevanti':  set[str]    # ground truth
                }

        Returns:
            Valore MAP in [0, 1]. 0.0 se lista vuota.
        """
        if not lista_risultati:
            return 0.0

        somma_ap = sum(
            self.average_precision(r['recuperati'], r['rilevanti'])
            for r in lista_risultati
        )
        return somma_ap / len(lista_risultati)

    def valuta_sistema(
        self,
        lista_risultati: list[dict],
        k_valori:        list[int] = None
    ) -> dict:
        """
        Calcola tutte le metriche IR su un insieme di query.

        Metodo di convenienza che calcola P@K, R@K, F1@K per
        ogni valore di K e il MAP, restituendo tutto in un
        unico dizionario pronto per la relazione.

        Args:
            lista_risultati: lista di dizionari con 'recuperati' e 'rilevanti'
            k_valori:        valori di K da calcolare (default: [1, 3, 5, 10])

        Returns:
            Dizionario con tutte le metriche:
            {
                'MAP': float,
                'P@1': float, 'R@1': float, 'F1@1': float,
                'P@3': float, 'R@3': float, 'F1@3': float,
                ...
            }
        """
        if k_valori is None:
            k_valori = [1, 3, 5, 10]

        n = len(lista_risultati)
        if n == 0:
            return {}

        risultato = {'MAP': self.mean_average_precision(lista_risultati)}

        for k in k_valori:
            somma_p  = sum(
                self.precision_at_k(r['recuperati'], r['rilevanti'], k)
                for r in lista_risultati
            )
            somma_r  = sum(
                self.recall_at_k(r['recuperati'], r['rilevanti'], k)
                for r in lista_risultati
            )
            somma_f1 = sum(
                self.f1_at_k(r['recuperati'], r['rilevanti'], k)
                for r in lista_risultati
            )
            risultato[f'P@{k}']  = round(somma_p  / n, 4)
            risultato[f'R@{k}']  = round(somma_r  / n, 4)
            risultato[f'F1@{k}'] = round(somma_f1 / n, 4)

        risultato['MAP'] = round(risultato['MAP'], 4)
        return risultato