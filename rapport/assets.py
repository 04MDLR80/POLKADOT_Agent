"""Génère les figures et tableaux LaTeX du rapport à partir des sorties du projet (outputs/)."""
import json, os, shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates
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

# ------------------------------------------------------------------ version 2 et direct
V2 = json.load(open(f"{OUT}/v2.json"))
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=False)
for ax, (e, titre) in zip(axs, (("val", "Validation (hausse puis rechute)"), ("test", "Test (chute de 75 %)"))):
    for k, col, lab in (("Buy and Hold", "#2F6FB5", "Buy and Hold"), ("v1", "#7A5CCF", "version 1"), ("v2", "#E6007A", "version 2")):
        c = V2["courbes"][e][k]
        ax.plot(pd.to_datetime(c["dates"]), c["valeur"], color=col, lw=1.6 if k == "v2" else 1.1, label=lab)
    ax.set_title(titre, fontsize=8.5); ax.tick_params(axis="y", labelsize=7); ax.tick_params(axis="x", labelsize=7)
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=3)); ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%y"))
axs[0].set_ylabel("Valeur du portefeuille ($)"); axs[1].legend(frameon=False, fontsize=7, loc="upper right")
fig.tight_layout(); fig.savefig(f"{F}/v2_courbes.pdf"); plt.close(fig)

t = "\\begin{tabular}{llrrrr}\\toprule\nPériode & Stratégie & Valeur finale (\\$) & Rendement & Drawdown max & Ordres \\\\\\midrule\n"
for e, nom_e in (("val", "Validation"), ("test", "Test")):
    for k, nom in (("v1", "Version 1"), ("v2", "Version 2"), ("Buy and Hold", "Buy and Hold")):
        r = V2["resume"][e][k]
        t += f"{nom_e if k == 'v1' else ''} & {nom} & {n0(r['valeur_finale'])} & {pc(r['rendement'])} & {pc(r['drawdown'])} & {r['ordres']} \\\\\n"
    t += "\\addlinespace\n" if e == "val" else ""
open(f"{ICI}/t_v2.tex", "w").write(t + "\\bottomrule\\end{tabular}\n")

t = "\\begin{tabular}{lrrrr}\\toprule\nZone de RSI (14 j) & Jours & Rendement prévu & Rendement réel & Signal d'achat \\\\\\midrule\n"
for r in V2["diagnostic_rsi"]:
    t += f"{r['zone']} & {r['n']} & {pc(r['prevu'], 2)} & {pc(r['reel'], 2)} & {r['part_achat']*100:.0f}\\,\\% \\\\\n"
open(f"{ICI}/t_rsi.tex", "w").write(t + "\\bottomrule\\end{tabular}\n")

LIVE = os.path.join(ICI, "..", "live", "etat_direct.json")
if os.path.exists(LIVE):
    L = json.load(open(LIVE))
    h = pd.DataFrame(L["histo"]); h["t"] = pd.to_datetime(h["t"])
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.plot(h.t, h.valeur, color="#E6007A", lw=1.6, label="agents")
    ax.plot(h.t, h.bh, color="#2F6FB5", lw=1.1, label="Buy and Hold")
    for d in L["decisions"]:
        if d["action"] != "RIEN":
            x = pd.to_datetime(d["execute_le"]).tz_convert(None) if pd.to_datetime(d["execute_le"]).tzinfo else pd.to_datetime(d["execute_le"])
            ax.axvline(x, color="#0B8A5A" if d["action"] == "ACHETER" else "#C2362B", lw=.8, ls=":")
    ax.axhline(8500, color="#999", lw=.7, ls="--"); ax.text(h.t.iloc[0], 8530, "seuil du coupe-circuit (8 500 $)", fontsize=7, color="#777")
    ax.set_ylabel("Valeur ($)"); ax.legend(frameon=False, fontsize=8, loc="lower left"); ax.tick_params(axis="x", labelsize=7)
    ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=3)); ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d/%m"))
    fig.tight_layout(); fig.savefig(f"{F}/direct.pdf"); plt.close(fig)
    t = "\\begin{tabular}{llrlr}\\toprule\nClôture du & Ordre & Prix (\\$) & Motif & Valeur après (\\$) \\\\\\midrule\n"
    motifs = {"achat validé (plafond 80%)": "achat validé", "stop-loss déclenché": "stop-loss", "coupe-circuit : liquidation": "coupe-circuit"}
    for d in L["decisions"]:
        if d["action"] != "RIEN":
            t += f"{pd.Timestamp(d['jour']):%d/%m/%Y} & {d['action'].lower()} & {d['prix']:.4f}".replace(".", ",") + \
                 f" & {motifs.get(d['raison'], d['raison'])} & {n0(d['valeur_apres'])} \\\\\n"
    open(f"{ICI}/t_direct.tex", "w").write(t + "\\bottomrule\\end{tabular}\n")
    D = dict(debut=L["debut"][:10], fin=h.t.iloc[-1].strftime("%d/%m/%Y"), valeur=L["histo"][-1]["valeur"], bh=L["histo"][-1]["bh"],
             prix0=L["prix_depart"], prix=L["prix"], n_dec=len(L["decisions"]),
             n_achat=sum(d["signal"] == "ACHAT" for d in L["decisions"]))
    json.dump(D, open(f"{ICI}/direct.json", "w"))
    print("direct", D)
print("v2 ok")
