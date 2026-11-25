"""Version 2 des agents : diagnostic et comparaison v1 / v2 sur la validation et le test.

Le choix de la v2 (filtre de tendance MM20 + coupe-circuit réarmable après 10 jours) est fait sur la VALIDATION ;
le test n'est lu qu'après. Utilise les prévisions déjà calculées par main.py (outputs/predictions.csv).
    python3 analyse_v2.py   ->  outputs/v2.json
"""
import json
import os

import numpy as np
import pandas as pd

import data_prep as dp
from agents import CONFIG, VERSION_2, RiskAgent, StrategyAgent, metrics, simulate

ICI = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ICI, "outputs")
base = dict(CONFIG)
base.update(json.load(open(f"{OUT}/metrics.json"))["configuration"])
base.update(filtre_tendance=None, pause_coupe_circuit=0)
v2 = dict(base, **VERSION_2)

raw = dp.load_raw(os.path.join(ICI, "data", "dot_usd_complet.csv"))
raw["Date"] = pd.to_datetime(raw["Date"])
f = dp.add_features(raw)
ma20 = raw["Close"].rolling(20).mean()
p = pd.read_csv(f"{OUT}/predictions.csv", parse_dates=["date_decision"])
idx = {d: i for i, d in enumerate(raw["Date"])}
P = {idx[r.date_decision]: (r.rendement_prevu, r.dispersion) for r in p.itertuples()}
jours = {e: [idx[d] for d in p[p.ensemble == e].date_decision] for e in ("val", "test")}


def signal(cfg):
    st = StrategyAgent(cfg)

    def fn(t, in_pos):
        mu, sg = P[t]
        return st.decide(mu, sg, in_pos, tendance_haussiere=bool(raw["Close"].iloc[t] > ma20.iloc[t])), dict(mu=mu, sigma=sg)
    return fn


def run(e, cfg, **kw):
    s = simulate(raw, jours[e], signal(cfg), RiskAgent(cfg, **kw), cfg=cfg, name="x")
    return s, metrics(s, cfg)


def bh(e):
    s = simulate(raw, jours[e], lambda t, ip: ("ACHAT" if not ip else "CONSERVATION", {}),
                 RiskAgent(base, cap=1.0, stop=0, circuit=0), cfg=base, name="bh")
    return s, metrics(s, base)


def ligne(m):
    return dict(valeur_finale=round(m["valeur_finale"], 2), rendement=m["rendement_total"], drawdown=m["max_drawdown"],
                sharpe=m["sharpe"], ordres=m["n_ordres"])


# 1. Grille évaluée sur la validation (filtre x coupe-circuit)
grille = []
for filtre in (None, "mm20"):
    for nom, pause, circuit in (("définitif (v1)", 0, None), ("réarmé après 10 j", 10, None), ("aucun", 0, 0)):
        cfg = dict(base, filtre_tendance=filtre, pause_coupe_circuit=pause)
        for e in ("val", "test"):
            _, m = run(e, cfg, circuit=circuit)
            grille.append(dict(ensemble=e, filtre="MM20" if filtre else "aucun", coupe_circuit=nom, **ligne(m)))

# 2. Diagnostic : prévision moyenne selon le RSI (le modèle parie sur un rebond quand le DOT est très survendu)
m_ = p.merge(f[["Date", "rsi14"]], left_on="date_decision", right_on="Date")
m_["zone"] = pd.qcut(m_["rsi14"], 5, labels=["très survendu", "survendu", "neutre", "acheté", "très acheté"])
diag = [dict(zone=str(z), n=int(len(g)), prevu=float(g.rendement_prevu.mean()), reel=float(g.rendement_reel.mean()),
             part_achat=float((g.rendement_prevu > base["seuil_achat"]).mean()))
        for z, g in m_.groupby("zone", observed=True)]

# 3. Courbes v1 / v2 / Buy and Hold
courbes = {}
for e in ("val", "test"):
    courbes[e] = {}
    for nom, cfg in (("v1", base), ("v2", v2)):
        s, _ = run(e, cfg)
        courbes[e][nom] = dict(dates=[str(pd.Timestamp(d).date()) for d in s["dates"]], valeur=[float(x) for x in s["equity"]])
    s, _ = bh(e)
    courbes[e]["Buy and Hold"] = dict(dates=[str(pd.Timestamp(d).date()) for d in s["dates"]], valeur=[float(x) for x in s["equity"]])

res = dict(version_2=VERSION_2, periodes={e: [str(raw["Date"].iloc[jours[e][0]].date()), str(raw["Date"].iloc[jours[e][-1] + 1].date())] for e in jours},
           resume={e: {"v1": ligne(run(e, base)[1]), "v2": ligne(run(e, v2)[1]), "Buy and Hold": ligne(bh(e)[1])} for e in jours},
           grille=grille, diagnostic_rsi=diag, courbes=courbes)
json.dump(res, open(f"{OUT}/v2.json", "w"), ensure_ascii=False, indent=1)
for e in jours:
    print(e, res["periodes"][e], {k: (v["valeur_finale"], round(v["rendement"] * 100, 1), round(v["drawdown"] * 100, 1), v["ordres"]) for k, v in res["resume"][e].items()})
print(pd.DataFrame(grille).assign(rendement=lambda d: (d.rendement * 100).round(1), drawdown=lambda d: (d.drawdown * 100).round(1)).to_string(index=False))
