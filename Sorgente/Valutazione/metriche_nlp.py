from collections import Counter

LABEL_FEVER = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']

class MetricheNLP:

    def accuracy(self, label_vere: list[str], label_predette: list[str]) -> float:
        if not label_vere:
            return 0.0
        corretti = sum(v == p for v, p in zip(label_vere, label_predette))
        return corretti / len(label_vere)

    def matrice_confusione(self, label_vere: list[str], label_predette: list[str], classi: list[str] = None) -> dict:
        if classi is None:
            classi = LABEL_FEVER
        matrice = {(v, p): 0 for v in classi for p in classi}
        for vera, predetta in zip(label_vere, label_predette):
            if (vera, predetta) in matrice:
                matrice[(vera, predetta)] += 1
        return matrice

    def metriche_per_classe(self, label_vere: list[str], label_predette: list[str], classi: list[str] = None) -> dict:
        if classi is None:
            classi = LABEL_FEVER
        conteggio_vere = Counter(label_vere)
        conteggio_predette = Counter(label_predette)
        risultati = {}
        for classe in classi:
            tp = sum(1 for v, p in zip(label_vere, label_predette) if v == classe and p == classe)
            fp = conteggio_predette[classe] - tp
            fn = conteggio_vere[classe] - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
            risultati[classe] = {
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
                'support': conteggio_vere[classe]
            }
        return risultati

    def f1_macro(self, label_vere: list[str], label_predette: list[str], classi: list[str] = None) -> float:
        if not label_vere:
            return 0.0
        metriche = self.metriche_per_classe(label_vere, label_predette, classi)
        f1_valori = [m['f1'] for m in metriche.values()]
        return round(sum(f1_valori) / len(f1_valori), 4)

    def kappa_cohen(self, label_vere: list[str], label_predette: list[str], classi: list[str] = None) -> float:
        if not label_vere:
            return 0.0
        if classi is None:
            classi = LABEL_FEVER
        n = len(label_vere)
        conteggio_vere = Counter(label_vere)
        conteggio_predette = Counter(label_predette)
        p_osservata = sum(v == p for v, p in zip(label_vere, label_predette)) / n
        p_caso = sum(
            (conteggio_vere[c] / n) * (conteggio_predette[c] / n)
            for c in classi
        )
        if p_caso == 1.0:
            return 0.0
        return round((p_osservata - p_caso) / (1 - p_caso), 4)

    def valuta_sistema(self, label_vere: list[str], label_predette: list[str], classi: list[str] = None) -> dict:
        if classi is None:
            classi = LABEL_FEVER
        return {
            'accuracy': round(self.accuracy(label_vere, label_predette), 4),
            'f1_macro': self.f1_macro(label_vere, label_predette, classi),
            'kappa_cohen': self.kappa_cohen(label_vere, label_predette, classi),
            'per_classe': self.metriche_per_classe(label_vere, label_predette, classi)
        }

    def stampa_report(self, label_vere: list[str], label_predette: list[str], classi: list[str] = None) -> None:
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