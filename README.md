# Projet Deep Learning GSTR S3 — Agents autonomes sur DOT/USD

Système d'investissement **simulé** (aucune transaction réelle) : prévision par LSTM (comparé au GRU),
quatre agents autonomes, moteur de portefeuille et registre blockchain.

## Lancer

```bash
python3 main.py data/dot_usd_complet.csv   # LSTM + GRU, agents, simulation, blockchain (~2 min)
python3 eda.py                             # analyse exploratoire, figures dans outputs/eda/
python3 build_dashboard.py                 # génère outputs/dashboard.html
python3 verify_chain.py outputs/ledger.json
```

Dépendances : `numpy`, `pandas`, `scipy`, `matplotlib` (aucun framework de Deep Learning requis).

## Fichiers

| Fichier | Rôle | Livrable |
|---|---|---|
| `data_prep.py` | Nettoyage, variables, fenêtres, découpage chronologique avec purge | 1, 2 |
| `eda.py` | Analyse exploratoire : statistiques, tests ADF / Jarque-Bera / Ljung-Box, 4 figures | 1, 5 |
| `gru_model.py` | LSTM et GRU en NumPy (BPTT, Adam, arrêt anticipé, test du gradient) | 1 |
| `agents.py` | Agents Prévision / Stratégie / Risque / Exécution, simulation, métriques | 1 |
| `blockchain.py`, `verify_chain.py` | Registre SHA-256 chaîné avec preuve de travail, vérification | 3 |
| `main.py` | Orchestration, choix LSTM/GRU sur la validation, références, ablation, sensibilité, exports | 1 |
| `dashboard_template.html`, `build_dashboard.py` | Tableau de bord (hors ligne) | 4 |
| `data/dot_usd_complet.csv` | Données brutes Investing.com (pDOTn), 08/02/2021 → 04/11/2026 | 2 |
| `outputs/base_donnees_preparee.csv` | Données + variables + cible + ensemble de chaque jour | 2 |
| `outputs/ledger.json` | Registre blockchain à partager avec le professeur | 3 |
| `outputs/predictions.csv`, `transactions.csv`, `metrics.json` | Résultats | 5 |

## Choix techniques

- **Variables** (jour t, connues à la clôture) : rendement log, amplitude, variation du volume, écart à la MM20, momentum 5 et 20 j, volatilité 20 j, RSI 14.
- **Cible** : rendement log cumulé sur 5 jours. Fenêtres de 20 jours.
- **Découpage** : chronologique 60 / 15 / 25 %, purge de 5 fenêtres aux frontières, normalisation sur le train seul.
- **Modèles** : ensembles de 5 LSTM et de 5 GRU ; hyperparamètres et famille choisis sur la validation. Retenu : LSTM, 4 unités, L2 = 0,01. Moyenne = prévision, écart-type = incertitude.
  PyTorch n'était pas disponible : les réseaux sont écrits en NumPy et leurs gradients vérifiés par différences finies (erreur relative < 1e-6).
- **Pas de fuite d'information** : décision sur la clôture de J, exécution à l'ouverture de J+1.
- **Coûts** : 0,26 % de frais + 0,05 % de glissement par ordre.
- **Stratégie** : achat si rendement prévu > 0,62 % (coût aller-retour) et > 0,5 × incertitude ; vente sous −0,56 % (40e centile des prévisions de validation).
- **Risque** : exposition max 80 %, stop-loss 10 %, coupe-circuit à 85 % du capital.
- **Références** : Buy and Hold, règle technique (clôture vs MM20), prévisions naïves.

## Résultats (test : 05/06/2025 → 31/10/2026, 513 jours)

| Stratégie | Valeur finale | Rendement | Drawdown max |
|---|---|---|---|
| Agents LSTM (retenu) | 8 187 $ | −18,1 % | −29,2 % |
| Agents GRU | 8 415 $ | −15,8 % | −23,3 % |
| Règle technique MM20 | 8 097 $ | −19,0 % | −30,4 % |
| Buy and Hold | 2 469 $ | −75,3 % | −83,5 % |
| Agents LSTM sans coupe-circuit (ablation) | 6 311 $ | −36,9 % | −54,6 % |

Sur le test, ni le LSTM (RMSE 0,0914) ni le GRU (0,0919) ne battent la prévision naïve « moyenne du train » (0,0887).
L'écart avec le Buy and Hold vient de la gestion du risque : après un stop-loss le 02/12/2025, le portefeuille
est passé sous le seuil du coupe-circuit, qui a bloqué les achats pendant la suite de la chute. Détails dans le rapport.

## Simulation en direct (paper trading)

Le modèle entraîné (`outputs/modele_live.npz`) est rechargé par `live_agent.py`, qui fait tourner les 4 agents
sur le prix réel de DOT/USD (API publique Kraken, sans compte ni clé). Aucun ordre réel n'est passé.

```bash
python live_agent.py            # direct : tableau de bord sur http://localhost:8050
python live_agent.py --rejeu    # rejeu accéléré de la période de test (même tableau de bord)
python live_agent.py --reset    # repartir de 10 000 $ en direct
```

Sous Windows : double-cliquer sur `lancer_direct.bat` ou `lancer_rejeu.bat`.

- Le prix et la valeur du portefeuille sont mis à jour toutes les 15 s.
- Les agents décident une fois par jour, à la clôture quotidienne (00:00 UTC), comme dans l'entraînement ;
  au premier lancement, ils décident tout de suite sur la dernière clôture.
- Chaque décision ajoute 4 blocs au registre `live/registre_direct.json`, vérifié à chaque mise à jour.
- L'état est sauvegardé dans `live/` : après un arrêt (Ctrl+C), le prochain lancement reprend là où il s'était arrêté.
