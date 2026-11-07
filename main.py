"""Pipeline complet : données -> GRU -> agents -> simulation -> blockchain -> métriques.

Usage : python3 main.py [chemin_csv]
Sorties dans outputs/ : base_donnees_preparee.csv, predictions.csv, transactions.csv,
ledger.json, metrics.json, results.json (pour le tableau de bord).
"""
import json
import sys

import numpy as np
import pandas as pd

from agents import CONFIG, ForecastAgent, RiskAgent, StrategyAgent, metrics, simulate
from blockchain import Blockchain
import data_prep
from data_prep import FEATURES, HORIZON, prepare
from gru_model import GRURegressor, LSTMRegressor, check_gradient

CSV = sys.argv[1] if len(sys.argv) > 1 else "data/dot_usd_complet.csv"
SEEDS = range(5)                    # ensemble de 5 GRU (graines différentes)
GRID = [dict(n_hid=4, l2=1e-2), dict(n_hid=8, l2=1e-2), dict(n_hid=8, l2=1e-3)]

np.random.seed(0)
P = prepare(CSV)
WINDOW = data_prep.WINDOW
raw, d, X, y, y_raw, idx, split = P["raw"], P["df"], P["X"], P["y"], P["y_raw"], P["idx"], P["split"]
y_sd = P["scaler"]["y_sd"]
tr, va, te = split == "train", split == "val", split == "test"

# ---------------- 1-2. LSTM et GRU : hyperparamètres choisis sur la validation, puis entraînement final ----------------
FAMILIES = {"GRU": GRURegressor, "LSTM": LSTMRegressor}


def ens_predict(ms, Xs):
    P_ = np.array([m.predict(Xs) for m in ms])
    return P_.mean(0) * y_sd, P_.std(0) * y_sd


fam = {}
for name, cls in FAMILIES.items():
    grid_res = []
    for g in GRID:
        ms = []
        for s in SEEDS:
            m = cls(len(FEATURES), g["n_hid"], seed=s)
            m.fit(X[tr], y[tr], X[va], y[va], l2=g["l2"])
            ms.append(m)
        rmse = float(np.sqrt(np.mean((ens_predict(ms, X[va])[0] - y_raw[va]) ** 2)))
        grid_res.append(dict(**g, rmse_val=rmse))
    best_g = min(grid_res, key=lambda r: r["rmse_val"])
    ms, hs = [], []
    for s in SEEDS:
        m = cls(len(FEATURES), best_g["n_hid"], seed=s)
        h, _ = m.fit(X[tr], y[tr], X[va], y[va], l2=best_g["l2"])
        ms.append(m)
        hs.append(h)
    mu_, sg_ = ens_predict(ms, X)
    fam[name] = dict(grid=grid_res, best=best_g, models=ms, hist=hs, pred=mu_, sig=sg_)
    print(name, "retenu :", best_g, flush=True)

# Le modèle utilisé par les agents = la famille de plus faible RMSE de validation (le test n'intervient pas)
MODEL = min(fam, key=lambda k: fam[k]["best"]["rmse_val"])
OTHER = "LSTM" if MODEL == "GRU" else "GRU"
grid_res, best = fam[MODEL]["grid"], fam[MODEL]["best"]
models, histories = fam[MODEL]["models"], fam[MODEL]["hist"]
pred_all, sig_all = fam[MODEL]["pred"], fam[MODEL]["sig"]

# Calibration des seuils de l'agent stratégie sur la VALIDATION (jamais sur le test) :
# achat si la prévision est dans les 30 % les plus hautes de la validation ET couvre les coûts aller-retour,
# vente si elle tombe sous la médiane basse (40e centile).
cout_ar = 2 * (CONFIG["frais"] + CONFIG["glissement"])
CONFIG["seuil_achat"] = float(max(np.quantile(pred_all[va], 0.70), cout_ar))
CONFIG["seuil_vente"] = float(np.quantile(pred_all[va], 0.40))
CONFIG["horizon_jours"] = HORIZON


def err(mask, p):
    e = p[mask] - y_raw[mask]
    return dict(MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())),
                direction=float(np.mean(np.sign(p[mask]) == np.sign(y_raw[mask]))))


naive_zero = np.zeros_like(y_raw)                      # référence : rendement nul
naive_mean = np.full_like(y_raw, y_raw[tr].mean())     # référence : moyenne du train
forecast_metrics = {s: {"GRU": err(m_, fam["GRU"]["pred"]), "LSTM": err(m_, fam["LSTM"]["pred"]),
                        "Naif_zero": err(m_, naive_zero),
                        "Naif_moyenne_train": err(m_, naive_mean)}
                    for s, m_ in (("train", tr), ("validation", va), ("test", te))}

# ---------------- 3. Agents + simulation sur la période de test ----------------
row_of = {t: i for i, t in enumerate(idx)}
test_days = [t for t in idx[te] if t + 1 < len(raw)]
fa = ForecastAgent(models, X, y_sd, row_of)
sa = StrategyAgent(CONFIG)


def model_signal(t, in_pos):
    mu, sg = fa.forecast(t)
    return sa.decide(mu, sg, in_pos), dict(mu=mu, sigma=sg)


ma20 = raw["Close"].rolling(20).mean()


def technical_signal(t, in_pos):          # règle technique : clôture au-dessus / en dessous de la MM20
    return ("ACHAT" if raw["Close"].iloc[t] > ma20.iloc[t] else "VENTE"), {}


def buy_hold_signal(t, in_pos):
    return ("ACHAT" if not in_pos else "CONSERVATION"), {}


AG = f"Agents {MODEL}"


def make_signal(pred, sig, cfg):
    """Signal d'un modèle à partir de ses prévisions (mêmes règles que l'agent stratégie)."""
    st = StrategyAgent(cfg)

    def fn(t, in_pos):
        mu, sg = float(pred[row_of[t]]), float(sig[row_of[t]])
        return st.decide(mu, sg, in_pos), dict(mu=mu, sigma=sg)
    return fn


ledger = Blockchain()
ledger.add_event("SYSTEME", "-", dict(
    actif="DOT/USD (Investing.com pDOTn, quotidien)", periode_donnees=[str(raw["Date"].min().date()), str(raw["Date"].max().date())],
    periode_test=[str(raw["Date"].iloc[test_days[0]].date()), str(raw["Date"].iloc[test_days[-1] + 1].date())],
    modele=f"ensemble de {len(models)} {MODEL} (fenêtre {WINDOW} j, horizon {HORIZON} j, {best['n_hid']} unités, l2={best['l2']})",
    configuration=CONFIG))
sim_agents = simulate(raw, test_days, model_signal, RiskAgent(CONFIG), ledger=ledger, name=AG)
sim_tech = simulate(raw, test_days, technical_signal, RiskAgent(CONFIG), name="Règle technique (MM20)")
sim_bh = simulate(raw, test_days, buy_hold_signal, RiskAgent(CONFIG, cap=1.0, stop=0, circuit=0), name="Buy and Hold")
M = [metrics(s) for s in (sim_agents, sim_tech, sim_bh)]
# Étude d'ablation (hors registre) : même modèle et mêmes règles, sans coupe-circuit, puis sans stop-loss ni coupe-circuit
sim_nocb = simulate(raw, test_days, model_signal, RiskAgent(CONFIG, circuit=0), name=f"{AG} sans coupe-circuit")
ablation = [metrics(sim_nocb),
            metrics(simulate(raw, test_days, model_signal, RiskAgent(CONFIG, stop=0, circuit=0), name=f"{AG} sans stop-loss ni coupe-circuit"))]
# Même système avec l'autre réseau (seuil de vente recalibré sur SA validation)
cfg_o = dict(CONFIG)
cfg_o["seuil_achat"] = float(max(np.quantile(fam[OTHER]["pred"][va], 0.70), cout_ar))
cfg_o["seuil_vente"] = float(np.quantile(fam[OTHER]["pred"][va], 0.40))
sim_autre = simulate(raw, test_days, make_signal(fam[OTHER]["pred"], fam[OTHER]["sig"], cfg_o),
                     RiskAgent(cfg_o), cfg=cfg_o, name=f"Agents {OTHER}")
autre = metrics(sim_autre)
autre["seuils"] = dict(achat=cfg_o["seuil_achat"], vente=cfg_o["seuil_vente"])

# Analyse de sensibilité (hors registre) : un paramètre modifié à la fois, le reste identique
sensi = []
base_sig = make_signal(pred_all, sig_all, CONFIG)
for label, param, val, kw in [
        ("Référence", "-", "-", {}),
        ("Coupe-circuit", "coupe_circuit", "aucun", dict(circuit=0)),
        ("Coupe-circuit", "coupe_circuit", "75 %", dict(circuit=0.75)),
        ("Coupe-circuit", "coupe_circuit", "90 %", dict(circuit=0.90)),
        ("Stop-loss", "stop_loss", "aucun", dict(stop=0)),
        ("Stop-loss", "stop_loss", "5 %", dict(stop=0.05)),
        ("Stop-loss", "stop_loss", "20 %", dict(stop=0.20)),
        ("Plafond d'exposition", "plafond_exposition", "50 %", dict(cap=0.5)),
        ("Plafond d'exposition", "plafond_exposition", "100 %", dict(cap=1.0))]:
    mm = metrics(simulate(raw, test_days, base_sig, RiskAgent(CONFIG, **kw), name=label))
    sensi.append(dict(parametre=label, valeur=val, **{k: mm[k] for k in ("valeur_finale", "rendement_total", "max_drawdown", "sharpe", "n_ordres")}))
for label, val, upd in [("Frais par ordre", "0,10 %", dict(frais=0.001)), ("Frais par ordre", "0,50 %", dict(frais=0.005)),
                        ("Seuil d'achat", "0,30 %", dict(seuil_achat=0.003)), ("Seuil d'achat", "1,50 %", dict(seuil_achat=0.015))]:
    cfg_s = {**CONFIG, **upd}
    mm = metrics(simulate(raw, test_days, make_signal(pred_all, sig_all, cfg_s), RiskAgent(cfg_s), cfg=cfg_s, name=label), cfg_s)
    sensi.append(dict(parametre=label, valeur=val, **{k: mm[k] for k in ("valeur_finale", "rendement_total", "max_drawdown", "sharpe", "n_ordres")}))

ledger.add_event("SYSTEME", "-", dict(resultats={m["strategie"]: dict(
    profit_net=round(m["profit_net"], 2), rendement_total=round(m["rendement_total"], 5),
    max_drawdown=round(m["max_drawdown"], 5)) for m in M}))
ok, msg = ledger.verify()
assert ok, msg

# ---------------- 4. Exports ----------------
pd.DataFrame(dict(date_decision=[raw["Date"].iloc[t].strftime("%Y-%m-%d") for t in idx],
                  date_cible=[raw["Date"].iloc[t + HORIZON].strftime("%Y-%m-%d") for t in idx],
                  ensemble=split, rendement_reel=y_raw, rendement_prevu=pred_all,
                  dispersion=sig_all, prevu_GRU=fam["GRU"]["pred"], prevu_LSTM=fam["LSTM"]["pred"])).to_csv("outputs/predictions.csv", index=False)
pd.DataFrame([{**l, "date_decision": l["date_decision"].strftime("%Y-%m-%d"),
               "date_execution": l["date_execution"].strftime("%Y-%m-%d")}
              for l in sim_agents["log"]]).to_csv("outputs/transactions.csv", index=False)
out = d.copy()
out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
out.to_csv("outputs/base_donnees_preparee.csv", index=False)
ledger.export("outputs/ledger.json")

# Modèle retenu, normalisation et règles : rechargés par live_agent.py pour la simulation en direct
np.savez("outputs/modele_live.npz",
         **{f"m{j}__{k}": v for j, m in enumerate(models) for k, v in m.p.items()},
         mu=P["scaler"]["mu"], sd=P["scaler"]["sd"], y_sd=np.array(y_sd),
         meta=np.array(json.dumps(dict(famille=MODEL, n_hid=best["n_hid"], n_modeles=len(models), fenetre=WINDOW,
                                       horizon=HORIZON, variables=FEATURES, config=CONFIG), ensure_ascii=False)))

summary = dict(
    donnees=dict(n_jours=len(raw), debut=str(raw["Date"].min().date()), fin=str(raw["Date"].max().date()),
                 n_fenetres=int(len(y)), n_train=int(tr.sum()), n_val=int(va.sum()), n_test=int(te.sum()),
                 fenetre=WINDOW, horizon=HORIZON, variables=FEATURES,
                 jours_ajoutes=raw.attrs.get("jours_ajoutes", [])),
    verification_gradient=dict(GRU=check_gradient(), LSTM=check_gradient(cls=LSTMRegressor)),
    modele_agents=MODEL, recherche_hyperparametres={k: v["grid"] for k, v in fam.items()},
    retenu=best, retenus={k: v["best"] for k, v in fam.items()},
    epoques={k: [len(h) for h in v["hist"]] for k, v in fam.items()},
    prevision=forecast_metrics, strategies=M, ablation=ablation, autre_modele=autre, sensibilite=sensi, blockchain=dict(valide=ok, message=msg, n_blocs=len(ledger.chain)),
    configuration=CONFIG)
json.dump(summary, open("outputs/metrics.json", "w"), ensure_ascii=False, indent=1)

results = dict(
    summary=summary,
    prix=dict(dates=raw["Date"].dt.strftime("%Y-%m-%d").tolist(), close=raw["Close"].tolist()),
    previsions=dict(dates=[raw["Date"].iloc[t].strftime("%Y-%m-%d") for t in idx], reel=y_raw.tolist(),
                    prevu=pred_all.tolist(), ensemble=split.tolist()),
    courbes={s["name"]: dict(dates=[x.strftime("%Y-%m-%d") for x in s["dates"]], valeur=s["equity"].tolist())
             for s in (sim_agents, sim_tech, sim_bh)},
    courbes_supp={s["name"]: dict(dates=[x.strftime("%Y-%m-%d") for x in s["dates"]], valeur=s["equity"].tolist())
                  for s in (sim_autre, sim_nocb)},
    apprentissage=[dict(train=[h[0] for h in hh], val=[h[1] for h in hh]) for hh in histories],
    transactions=[{k: (v.strftime("%Y-%m-%d") if hasattr(v, "strftime") else v) for k, v in l.items()}
                  for l in sim_agents["log"]],
    ledger=ledger.chain)
json.dump(results, open("outputs/results.json", "w"), ensure_ascii=False)

print(json.dumps(dict(retenu=best, prevision_test=forecast_metrics["test"],
                      strategies=[{k: m[k] for k in ("strategie", "profit_net", "rendement_total", "max_drawdown", "sharpe", "n_ordres")} for m in M + ablation + [autre]],
                      sensibilite=sensi,
                      blockchain=msg), ensure_ascii=False, indent=1))
