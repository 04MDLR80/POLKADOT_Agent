"""Génère les figures et tableaux LaTeX du rapport à partir des sorties du projet (outputs/)."""
import json, os, shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ICI = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ICI, "..", "outputs"); F = os.path.join(ICI, "figs")
res = json.load(open(f"{OUT}/results.json")); met = json.load(open(f"{OUT}/metrics.json"))
pred = pd.read_csv(f"{OUT}/predictions.csv", parse_dates=["date_decision"]); tr = pd.read_csv(f"{OUT}/transactions.csv", parse_dates=["date_execution"])
eda = json.load(open(f"{OUT}/eda/eda.json"))
C = {"Agents LSTM": "#E6007A", "Buy and Hold": "#2F6FB5", "Règle technique (MM20)": "#B87800", "Agents GRU": "#7A5CCF",
     "Agents LSTM sans coupe-circuit": "#00907F"}
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#E6E8EC", "grid.linewidth": .6, "axes.axisbelow": True, "figure.dpi": 150})
for f in os.listdir(f"{OUT}/eda"):
    if f.endswith(".png"): shutil.copy(f"{OUT}/eda/{f}", F)

# 1. Courbes de valeur
fig, ax = plt.subplots(figsize=(6.4, 3.0))
cur = {**res["courbes"], **res["courbes_supp"]}
for n in ["Buy and Hold", "Règle technique (MM20)", "Agents GRU", "Agents LSTM sans coupe-circuit", "Agents LSTM"]:
    s = pd.Series(cur[n]["valeur"], index=pd.to_datetime(cur[n]["dates"]))
    ax.plot(s.index, s.values, color=C[n], lw=1.8 if n == "Agents LSTM" else 1.2, ls="--" if "sans" in n else "-", label=n)
ax.set_ylabel("Valeur du portefeuille ($)"); ax.legend(frameon=False, fontsize=7.5, loc="upper right"); fig.tight_layout()
fig.savefig(f"{F}/valeur.pdf"); plt.close(fig)

# 2. Prix + ordres + valeur
px = pd.Series(res["prix"]["close"], index=pd.to_datetime(res["prix"]["dates"]))
d0 = pd.Timestamp(pred.loc[pred.ensemble == "test", "date_decision"].min())
pt = px[px.index >= d0]
fig, ax = plt.subplots(figsize=(6.4, 2.8)); ax.plot(pt.index, pt.values, color="#555", lw=1.1)
a = tr[tr.action == "ACHETER"]; v = tr[tr.action == "VENDRE"]
ax.scatter(a.date_execution, a.fill, marker="^", color="#0B8A5A", s=28, zorder=3, label="achat")
ax.scatter(v.date_execution, v.fill, marker="v", color="#C2362B", s=28, zorder=3, label="vente")
ax.axvline(pd.Timestamp("2025-12-02"), color="#E6007A", lw=1, ls=":"); ax.text(pd.Timestamp("2025-12-08"), pt.max() * .92, "coupe-circuit", color="#E6007A", fontsize=7.5)
ax.set_ylabel("DOT/USD ($)"); ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(f"{F}/ordres.pdf"); plt.close(fig)

# 3. Prévu vs réel (test)
p = pred[pred.ensemble == "test"].set_index("date_decision")
fig, ax = plt.subplots(figsize=(6.4, 2.6))
ax.plot(p.index, p.rendement_reel * 100, color="#2F6FB5", lw=1, label="rendement réel à 5 jours")
ax.plot(p.index, p.prevu_LSTM * 100, color="#E6007A", lw=1.4, label="prévu (LSTM)")
ax.axhline(0, color="#999", lw=.6); ax.set_ylabel("%"); ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(f"{F}/previsions.pdf"); plt.close(fig)

# 4. Apprentissage
fig, ax = plt.subplots(figsize=(6.4, 2.4))
for i, a_ in enumerate(res["apprentissage"]):
    ax.plot(a_["train"], color="#2F6FB5", lw=.9, alpha=.8, label="entraînement" if i == 0 else None)
    ax.plot(a_["val"], color="#E6007A", lw=.9, alpha=.8, label="validation" if i == 0 else None)
ax.set_xlabel("époque"); ax.set_ylabel("erreur quadratique (normalisée)"); ax.legend(frameon=False, fontsize=8); fig.tight_layout()
fig.savefig(f"{F}/apprentissage.pdf"); plt.close(fig)

# --- tableaux
def pc(x, d=1): return f"{x*100:+.{d}f}".replace(".", ",").replace("-", "$-$") + "\\,\\%"
def n0(x): return f"{x:,.0f}".replace(",", "\\,")
S = {s["strategie"]: s for s in met["strategies"] + met["ablation"] + [met["autre_modele"]]}
rows = [("Agents LSTM (retenu)", S["Agents LSTM"]), ("Agents GRU", S["Agents GRU"]), ("Règle technique MM20", S["Règle technique (MM20)"]),
        ("Buy and Hold", S["Buy and Hold"]), ("Agents LSTM sans coupe-circuit", S["Agents LSTM sans coupe-circuit"])]
t = "\\begin{tabular}{lrrrrr}\\toprule\nStratégie & Valeur finale (\\$) & Rendement & Drawdown max & Sharpe & Ordres \\\\\\midrule\n"
for n, s in rows: t += f"{n} & {n0(s['valeur_finale'])} & {pc(s['rendement_total'])} & {pc(s['max_drawdown'])} & {('%.2f' % s['sharpe']).replace('.',',').replace('-','$-$')} & {s['n_ordres']} \\\\\n"
open(f"{ICI}/t_strategies.tex", "w").write(t + "\\bottomrule\\end{tabular}\n")

P = met["prevision"]; noms = {"LSTM": "LSTM", "GRU": "GRU", "Naif_zero": "Prévision nulle", "Naif_moyenne_train": "Moyenne du train"}
t = "\\begin{tabular}{lrrrrrr}\\toprule\n & \\multicolumn{3}{c}{Validation} & \\multicolumn{3}{c}{Test} \\\\\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\nModèle & MAE & RMSE & Direction & MAE & RMSE & Direction \\\\\\midrule\n"
f4 = lambda x: f"{x:.4f}".replace(".", ","); fp = lambda x: f"{x*100:.0f}\\,\\%"
for k in ["LSTM", "GRU", "Naif_zero", "Naif_moyenne_train"]:
    v, te = P["validation"][k], P["test"][k]
    dv = "--" if k == "Naif_zero" else fp(v["direction"]); dt = "--" if k == "Naif_zero" else fp(te["direction"])
    t += f"{noms[k]} & {f4(v['MAE'])} & {f4(v['RMSE'])} & {dv} & {f4(te['MAE'])} & {f4(te['RMSE'])} & {dt} \\\\\n"
open(f"{ICI}/t_prevision.tex", "w").write(t + "\\bottomrule\\end{tabular}\n")

t = "\\begin{tabular}{llrrr}\\toprule\nParamètre & Valeur & Valeur finale (\\$) & Rendement & Drawdown max \\\\\\midrule\n"
for r in met["sensibilite"]:
    t += f"{r['parametre']} & {str(r['valeur']).replace('%','\\,\\%')} & {n0(r['valeur_finale'])} & {pc(r['rendement_total'])} & {pc(r['max_drawdown'])} \\\\\n"
open(f"{ICI}/t_sensibilite.tex", "w").write(t.replace("\\% \\%", "\\%") + "\\bottomrule\\end{tabular}\n")

H = met["recherche_hyperparametres"]
t = "\\begin{tabular}{lrrr}\\toprule\nFamille & Unités cachées & Pénalité L2 & RMSE de validation \\\\\\midrule\n"
for fam, L in H.items():
    for r in L: t += f"{fam} & {r['n_hid']} & {str(r['l2']).replace('.',',')} & {f4(r['rmse_val'])} \\\\\n"
open(f"{ICI}/t_hyper.tex", "w").write(t + "\\bottomrule\\end{tabular}\n")

t = "\\begin{tabular}{rrrr}\\toprule\nAnnée & Rendement & Volatilité annualisée & Jours \\\\\\midrule\n"
for r in eda["par_annee"]: t += f"{r['annee']} & {pc(r['rendement'],0)} & {pc(r['volatilite'],0).replace('+','')} & {r['n_jours']} \\\\\n"
open(f"{ICI}/t_annees.tex", "w").write(t + "\\bottomrule\\end{tabular}\n")

t = "\\begin{tabular}{lrrl}\\toprule\nDate d'exécution & Ordre & Prix (\\$) & Motif \\\\\\midrule\n"
for _, r in tr[tr.action != "RIEN"].iterrows():
    mot = {"achat validé (plafond 80%)": "achat validé", "signal de vente validé": "signal de vente", "stop-loss déclenché": "stop-loss"}.get(r.reason, r.reason)
    t += f"{r.date_execution:%d/%m/%Y} & {r.action.lower()} & {r.fill:.3f}".replace(".", ",") + f" & {mot} \\\\\n"
open(f"{ICI}/t_ordres.tex", "w").write(t + "\\bottomrule\\end{tabular}\n")
print("ok")
