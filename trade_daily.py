"""Passage autonome des agents (paper trading) — exécuté par GitHub Actions (voir .github/workflows/trade.yml).

À chaque passage : prix DOT/USD (API publique Kraken) -> valorisation ; si une nouvelle bougie quotidienne
est clôturée, les 4 agents décident, l'ordre simulé est exécuté au prix courant et 4 blocs sont ajoutés au
registre. L'état est écrit dans live/ (versionné par git) et lu par le tableau de bord Streamlit.
Aucun ordre réel n'est jamais passé.
"""
import sys

import live_agent as la


def main():
    sim = la.Simulateur(la.Modele(), "direct")
    prix = la.kraken_prix()
    sim.maj_prix(prix)
    d = sim.decider(la.kraken_bougies(), prix)
    sim.statut, sim.erreur = "autonome (GitHub Actions)", None
    sim.sauver()
    if d:
        print(f"Décision sur la clôture du {d['jour']} : prévu {d['mu'] * 100:+.2f} %, signal {d['signal']}, "
              f"{d['action']} ({d['raison']}) -> valeur {d['valeur_apres']:.2f} $")
    else:
        print("Pas de nouvelle bougie clôturée : valorisation seulement.")
    print(f"Prix {prix:.4f} $ · agents {sim.pf.value(prix):.2f} $ · {len(sim.reg.chain)} blocs")


if __name__ == "__main__":
    sys.exit(main())
