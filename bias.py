import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

class GenderBiasSimulation:
    def __init__(self, 
                 bias=0.05, 
                 years=12, 
                 seed=43, 
                 n_girls=500, 
                 n_boys=500, 
                 max_total=600):
        self.bias = bias
        self.years = years
        self.seed = seed
        self.n_girls = n_girls
        self.n_boys = n_boys
        self.max_girls = max_total
        self.max_boys = max_total
        self.giorni = self.years * 365
        self.calo_giornaliero = (1 - self.bias * 1.5) ** (1 / 365)
        self.rumore_std = 0.0001

        # Containers
        self.interessate_ragazze = [n_girls]
        self.interessate_ragazze_nobias = [n_girls]
        self.interessati_ragazzi = [n_boys]

        self.percentuali_ragazze = [n_girls / self.max_girls * 100]
        self.percentuali_ragazze_nobias = [n_girls / self.max_girls * 100]
        self.percentuali_ragazzi = [n_boys / self.max_boys * 100]

    def run(self):
        np.random.seed(self.seed)

        ragazze = self.n_girls
        ragazze_nobias = self.n_girls
        ragazzi = self.n_boys

        for _ in range(self.giorni):
            rumore_ragazze = np.random.normal(0, self.rumore_std)
            ragazze *= self.calo_giornaliero * (1 + rumore_ragazze)
            ragazze = max(0, min(ragazze, self.max_girls))

            # Same rumore for ragazze_nobias
            ragazze_nobias *= (1 + rumore_ragazze)
            ragazze_nobias = max(0, min(ragazze_nobias, self.max_girls))

            rumore_ragazzi = np.random.normal(0, self.rumore_std)
            ragazzi *= (1 + rumore_ragazzi)
            ragazzi = max(0, min(ragazzi, self.max_boys))

            self.interessate_ragazze.append(ragazze)
            self.interessate_ragazze_nobias.append(ragazze_nobias)
            self.interessati_ragazzi.append(ragazzi)

            self.percentuali_ragazze.append(ragazze / self.max_girls * 100)
            self.percentuali_ragazze_nobias.append(ragazze_nobias / self.max_girls * 100)
            self.percentuali_ragazzi.append(ragazzi / self.max_boys * 100)

    def plot_and_save(self):
        giorni_lista = list(range(self.giorni + 1))
        anni_labels = [i for i in range(self.years + 1)]
        anni_posizioni = [i * 365 for i in anni_labels]

        # Print results
        print("== RISULTATI FINALI DOPO {} ANNI ==".format(self.years))
        print(f"Ragazze con bias:     {self.interessate_ragazze[-1]:.0f} su {self.max_girls} "
              f"({self.percentuali_ragazze[-1]:.1f}%)")
        print(f"Ragazze senza bias:   {self.interessate_ragazze_nobias[-1]:.0f} su {self.max_girls} "
              f"({self.percentuali_ragazze_nobias[-1]:.1f}%)")
        print(f"Ragazzi:              {self.interessati_ragazzi[-1]:.0f} su {self.max_boys} "
              f"({self.percentuali_ragazzi[-1]:.1f}%)")

        # Plot
        plt.figure(figsize=(12, 5))

        # Absolute values
        plt.subplot(1, 2, 1)
        plt.plot(giorni_lista, self.interessate_ragazze, label='Ragazze (con bias)', color='blue')
        plt.plot(giorni_lista, self.interessate_ragazze_nobias, label='Ragazze (senza bias)', color='violet', linestyle='--')
        plt.plot(giorni_lista, self.interessati_ragazzi, label='Ragazzi', color='orange')
        plt.axhline(self.max_girls, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.xticks(anni_posizioni, anni_labels)
        plt.xlabel('Anni')
        plt.ylabel('Numero di studenti interessati')
        plt.title(f'Interesse nella fisica (valori assoluti, bias = {self.bias*100:.1f}%)')
        plt.legend()
        plt.grid(True)

        # Percentuali
        plt.subplot(1, 2, 2)
        plt.plot(giorni_lista, self.percentuali_ragazze, label='Ragazze (con bias)', color='blue')
        plt.plot(giorni_lista, self.percentuali_ragazze_nobias, label='Ragazze (senza bias)', color='violet', linestyle='--')
        plt.plot(giorni_lista, self.percentuali_ragazzi, label='Ragazzi', color='orange')
        plt.xticks(anni_posizioni, anni_labels)
        plt.xlabel('Anni')
        plt.ylabel('Percentuale rispetto al massimo')
        plt.title(f'Interesse nella fisica (bias = {self.bias*100:.1f}%)')
        plt.ylim(0, 105)
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        # Save to PNG
        filename = (
            f"sim_bias{int(self.bias*1000)}_years{self.years}_"
            f"seed{self.seed}_girls{self.n_girls}_boys{self.n_boys}.png"
        )
        plt.savefig(filename, dpi=300)
        print(f"\nGrafico salvato come '{filename}'")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simula il bias di genere nell'interesse per la fisica.")

    parser.add_argument('--bias', type=float, default=0.05, help='Bias annuo come decimale (es. 0.05 = 5%)')
    parser.add_argument('--years', type=int, default=12, help='Numero di anni da simulare')
    parser.add_argument('--seed', type=int, default=43, help='Seed per la generazione casuale')
    parser.add_argument('--girls', type=int, default=500, help='Numero iniziale di ragazze')
    parser.add_argument('--boys', type=int, default=500, help='Numero iniziale di ragazzi')
    parser.add_argument('--max', type=int, default=600, help='Numero massimo di studenti interessati')

    args = parser.parse_args()

    sim = GenderBiasSimulation(
        bias=args.bias,
        years=args.years,
        seed=args.seed,
        n_girls=args.girls,
        n_boys=args.boys,
        max_total=args.max
    )
    sim.run()
    sim.plot_and_save()
