"""
metriche_nlp.py
════════════════════════════════════════════════════════════════════
Metriche di valutazione per il modulo di classificazione NLI.

Responsabilità:
    Calcolare le metriche standard di classificazione per valutare
    la qualità del classificatore NLI rispetto al ground truth FEVER.

Metriche implementate:

    Accuracy:
        Frazione di predizioni corrette sul totale.
        accuracy = corretti / totale
        Attenzione: ingannevole su dataset sbilanciati come FEVER
        (62% SUPPORTS → un classificatore che predice sempre SUPPORTS
        ottiene 62% di accuracy senza capire nulla).

    Precision per classe:
        Tra le predizioni di una classe, quante sono corrette.
        precision(C) = TP(C) / (TP(C) + FP(C))
        Misura quante volte il sistema è affidabile quando dice "C".

    Recall per classe:
        Tra le istanze vere di una classe, quante vengono trovate.
        recall(C) = TP(C) / (TP(C) + FN(C))
        Misura quante istanze della classe il sistema non perde.

    F1 per classe:
        Media armonica di precision e recall per classe.
        F1(C) = 2 * P(C) * R(C) / (P(C) + R(C))

    F1 macro:
        Media degli F1 di tutte le classi senza ponderazione.
        Tratta ogni classe ugualmente — appropriato per FEVER
        dove vogliamo performance bilanciate su tutte e tre le label.

    Kappa di Cohen:
        Misura l'accordo tra predizioni e ground truth, corretto
        per l'accordo atteso per caso.
        κ = (P_o - P_e) / (1 - P_e)
        dove P_o = accuracy osservata, P_e = accuracy attesa per caso.

        Interpretazione:
            κ < 0.20  → accordo scarso
            κ 0.21-0.40 → accordo discreto
            κ 0.41-0.60 → accordo moderato
            κ 0.61-0.80 → accordo sostanziale
            κ > 0.80  → accordo quasi perfetto

        Preferito all'accuracy perché non è influenzato dallo
        sbilanciamento delle classi nel dataset FEVER.

Utilizzo:
    from sorgente.valutazione.metriche_nlp import MetricheNLP

    metriche = MetricheNLP()
    risultati = metriche.valuta_sistema(label_vere, label_predette)
════════════════════════════════════════════════════════════════════
"""

from collections import Counter


# Label del dataset FEVER nell'ordine standard
LABEL_FEVER = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']


class MetricheNLP:
    """
    Calcola le metriche di valutazione per il classificatore NLI.

    Tutte le metriche operano su liste parallele di label:
        label_vere:     ground truth del dataset FEVER
        label_predette: output del classificatore NLI
    """

    # ── Metriche base ─────────────────────────────────────────────

    def accuracy(
        self,
        label_vere:     list[str],
        label_predette: list[str]
    ) -> float:
        """
        Calcola l'accuracy — frazione di predizioni corrette.

        accuracy = numero_corretti / totale

        Attenzione: su FEVER (62% SUPPORTS) l'accuracy può essere
        fuorviante. Un classificatore naive che predice sempre
        SUPPORTS ottiene ~62% senza capire nulla del testo.
        Usare sempre in combinazione con F1 macro e kappa.

        Args:
            label_vere:     lista di label ground truth
            label_predette: lista di label predette

        Returns:
            Accuracy in [0, 1]. 0.0 se lista vuota.
        """
        if not label_vere:
            return 0.0
        corretti = sum(v == p for v, p in zip(label_vere, label_predette))
        return corretti / len(label_vere)

    def matrice_confusione(
        self,
        label_vere:     list[str],
        label_predette: list[str],
        classi:         list[str] = None
    ) -> dict:
        """
        Calcola la matrice di confusione.

        La matrice mostra per ogni classe vera (riga) quante
        istanze sono state predette in ogni classe (colonna).
        La diagonale principale contiene le predizioni corrette.

        Args:
            label_vere:     lista di label ground truth
            label_predette: lista di label predette
            classi:         lista di classi (default: LABEL_FEVER)

        Returns:
            Dizionario {(vera, predetta): conteggio}
        """
        if classi is None:
            classi = LABEL_FEVER

        matrice = {(v, p): 0 for v in classi for p in classi}
        for vera, predetta in zip(label_vere, label_predette):
            if (vera, predetta) in matrice:
                matrice[(vera, predetta)] += 1
        return matrice

    def metriche_per_classe(
        self,
        label_vere:     list[str],
        label_predette: list[str],
        classi:         list[str] = None
    ) -> dict:
        """
        Calcola precision, recall e F1 per ogni classe.

        Per ogni classe C:
            TP = predetti C e sono C
            FP = predetti C ma non sono C
            FN = sono C ma predetti diversamente

            precision(C) = TP / (TP + FP)
            recall(C)    = TP / (TP + FN)
            F1(C)        = 2 * P * R / (P + R)

        Args:
            label_vere:     lista di label ground truth
            label_predette: lista di label predette
            classi:         liste di classi (default: LABEL_FEVER)

        Returns:
            {classe: {'precision': float, 'recall': float, 'f1': float, 'support': int}}
        """
        if classi is None:
            classi = LABEL_FEVER

        conteggio_vere     = Counter(label_vere)
        conteggio_predette = Counter(label_predette)

        risultati = {}
        for classe in classi:
            tp = sum(1 for v, p in zip(label_vere, label_predette)
                     if v == classe and p == classe)
            fp = conteggio_predette[classe] - tp
            fn = conteggio_vere[classe] - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = (2 * precision * recall / (precision + recall)
                         if (precision + recall) > 0 else 0.0)

            risultati[classe] = {
                'precision': round(precision, 4),
                'recall':    round(recall,    4),
                'f1':        round(f1,        4),
                'support':   conteggio_vere[classe]
            }
        return risultati

    def f1_macro(
        self,
        label_vere:     list[str],
        label_predette: list[str],
        classi:         list[str] = None
    ) -> float:
        """
        Calcola l'F1 macro — media degli F1 per classe.

        F1_macro = (1/|C|) * Σ F1(c)  per ogni classe c

        Tratta ogni classe ugualmente indipendentemente dalla
        frequenza nel dataset. Appropriato per FEVER dove vogliamo
        performance bilanciate su SUPPORTS, REFUTES e NOT ENOUGH INFO,
        anche se SUPPORTS è la classe dominante (55%).

        Args:
            label_vere:     lista di label ground truth
            label_predette: lista di label predette
            classi:         lista di classi (default: LABEL_FEVER)

        Returns:
            F1 macro in [0, 1]. 0.0 se lista vuota.
        """
        if not label_vere:
            return 0.0

        metriche = self.metriche_per_classe(label_vere, label_predette, classi)
        f1_valori = [m['f1'] for m in metriche.values()]
        return round(sum(f1_valori) / len(f1_valori), 4)

    def kappa_cohen(
        self,
        label_vere:     list[str],
        label_predette: list[str],
        classi:         list[str] = None
    ) -> float:
        """
        Calcola il Kappa di Cohen.

        Misura l'accordo tra predizioni e ground truth corretto
        per l'accordo atteso per puro caso.

        κ = (P_o - P_e) / (1 - P_e)

        dove:
            P_o = accuracy osservata (accordo reale)
            P_e = probabilità di accordo per caso
                = Σ P(vera=C) * P(predetta=C)

        Perché è migliore dell'accuracy su FEVER:
            Su FEVER (55% SUPPORTS), un classificatore che predice
            sempre SUPPORTS ha accuracy=0.55 ma κ≈0 (nessun accordo
            reale oltre il caso). Il kappa smonta l'illusione
            dell'accuracy su dataset sbilanciati.

        Interpretazione:
            κ < 0.20   → accordo scarso
            κ 0.21-0.40 → accordo discreto
            κ 0.41-0.60 → accordo moderato     ← risultato notebook
            κ 0.61-0.80 → accordo sostanziale
            κ > 0.80   → accordo quasi perfetto

        Args:
            label_vere:     lista di label ground truth
            label_predette: lista di label predette
            classi:         lista di classi (default: LABEL_FEVER)

        Returns:
            Kappa in [-1, 1]. 0.0 se lista vuota o accordo pari al caso.
        """
        if not label_vere:
            return 0.0

        if classi is None:
            classi = LABEL_FEVER

        n = len(label_vere)
        conteggio_vere     = Counter(label_vere)
        conteggio_predette = Counter(label_predette)

        # Accuratezza osservata
        p_osservata = sum(v == p for v, p in zip(label_vere, label_predette)) / n

        # Accuratezza attesa per caso
        p_caso = sum(
            (conteggio_vere[c] / n) * (conteggio_predette[c] / n)
            for c in classi
        )

        if p_caso == 1.0:
            return 0.0

        return round((p_osservata - p_caso) / (1 - p_caso), 4)

    # ── Metodo aggregato ──────────────────────────────────────────

    def valuta_sistema(
        self,
        label_vere:     list[str],
        label_predette: list[str],
        classi:         list[str] = None
    ) -> dict:
        """
        Calcola tutte le metriche NLP in un unico dizionario.

        Metodo di convenienza per la valutazione finale —
        produce tutti i numeri necessari per la relazione.

        Args:
            label_vere:     lista di label ground truth
            label_predette: lista di label predette
            classi:         lista di classi (default: LABEL_FEVER)

        Returns:
            Dizionario con tutte le metriche:
            {
                'accuracy':      float,
                'f1_macro':      float,
                'kappa_cohen':   float,
                'per_classe': {
                    'SUPPORTS':        {'precision', 'recall', 'f1', 'support'},
                    'REFUTES':         {'precision', 'recall', 'f1', 'support'},
                    'NOT ENOUGH INFO': {'precision', 'recall', 'f1', 'support'}
                }
            }
        """
        if classi is None:
            classi = LABEL_FEVER

        return {
            'accuracy':    round(self.accuracy(label_vere, label_predette), 4),
            'f1_macro':    self.f1_macro(label_vere, label_predette, classi),
            'kappa_cohen': self.kappa_cohen(label_vere, label_predette, classi),
            'per_classe':  self.metriche_per_classe(label_vere, label_predette, classi)
        }

    def stampa_report(
        self,
        label_vere:     list[str],
        label_predette: list[str],
        classi:         list[str] = None
    ) -> None:
        """
        Stampa un report formattato di tutte le metriche.

        Formato simile a sklearn classification_report ma
        con l'aggiunta del Kappa di Cohen.

        Args:
            label_vere:     lista di label ground truth
            label_predette: lista di label predette
            classi:         lista di classi (default: LABEL_FEVER)
        """
        risultati = self.valuta_sistema(label_vere, label_predette, classi)

        print('REPORT VALUTAZIONE NLI')
        print('=' * 60)
        print(f"{'Classe':<22} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
        print('-' * 60)
        for classe, m in risultati['per_classe'].items():
            print(f"  {classe:<20} {m['precision']:>10.4f} {m['recall']:>8.4f} "
                  f"{m['f1']:>8.4f} {m['support']:>9}")
        print('-' * 60)
        print(f"  {'Accuracy':<20} {risultati['accuracy']:>10.4f}")
        print(f"  {'F1 Macro':<20} {risultati['f1_macro']:>10.4f}")
        print(f"  {'Kappa di Cohen':<20} {risultati['kappa_cohen']:>10.4f}")

        kappa = risultati['kappa_cohen']
        if kappa > 0.80:
            interp = 'accordo quasi perfetto'
        elif kappa > 0.60:
            interp = 'accordo sostanziale'
        elif kappa > 0.40:
            interp = 'accordo moderato'
        elif kappa > 0.20:
            interp = 'accordo discreto'
        else:
            interp = 'accordo scarso'
        print(f"  → Kappa: {interp}")