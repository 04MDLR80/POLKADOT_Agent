"""Étape 1 - Analyse exploratoire : statistiques, tests, figures (outputs/eda/)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from data_prep import load_raw

OUT = "outputs/eda"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "figure.dpi": 150})
INK, ACC, GREY = "#222222", "#1f6fd1", "#9a9a9a"

raw = load_raw("data/dot_usd_complet.csv")
d = raw.set_index("Date")
r = np.log(d["Close"]).diff().dropna()


def adf_test(x, lags=1):
    """Test de Dickey-Fuller augmenté (constante, `lags` retards), statistique t sur gamma."""
    x = np.asarray(x, float)
    dx = np.diff(x)
    Y = dx[lags:]
    cols = [np.ones(len(Y)), x[lags:-1]] + [dx[lags - k:-k] for k in range(1, lags + 1)]
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    res = Y - X @ beta
    s2 = res @ res / (len(Y) - X.shape[1])
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    return float(beta[1] / se)


def acf(x, nlags=30):
    x = np.asarray(x) - np.mean(x)
    c0 = x @ x
    return np.array([x[:-k] @ x[k:] / c0 for k in range(1, nlags + 1)])


def ljung_box(x, lags=10):
    n = len(x)
    rho = acf(x, lags)
    q = n * (n + 2) * np.sum(rho ** 2 / (n - np.arange(1, lags + 1)))
    return float(q), float(1 - stats.chi2.cdf(q, lags))


# ---------- statistiques descriptives ----------
peak = d["Close"].cummax()
jb = stats.jarque_bera(r)
lb_r, p_r = ljung_box(r.values, 10)
lb_r2, p_r2 = ljung_box((r ** 2).values, 10)
summary = dict(
    n_jours=len(d), debut=str(d.index.min().date()), fin=str(d.index.max().date()),
    prix_min=float(d["Close"].min()), date_min=str(d["Close"].idxmin().date()),
    prix_max=float(d["Close"].max()), date_max=str(d["Close"].idxmax().date()),
    prix_debut=float(d["Close"].iloc[0]), prix_fin=float(d["Close"].iloc[-1]),
    perte_max_depuis_sommet=float((d["Close"] / peak - 1).min()),
    rendement_moyen_jour=float(r.mean()), ecart_type_jour=float(r.std()),
    volatilite_annualisee=float(r.std() * np.sqrt(365)),
    asymetrie=float(stats.skew(r)), kurtosis_excess=float(stats.kurtosis(r)),
    pire_jour=float(r.min()), date_pire=str(r.idxmin().date()),
    meilleur_jour=float(r.max()), date_meilleur=str(r.idxmax().date()),
    part_jours_hausse=float((r > 0).mean()),
    jarque_bera=float(jb.statistic), jarque_bera_p=float(jb.pvalue),
    adf_prix_log=adf_test(np.log(d["Close"].values)), adf_rendements=adf_test(r.values),
    adf_valeur_critique_5pc=-2.86,
    ljung_box_rendements_q10=lb_r, ljung_box_rendements_p=p_r,
    ljung_box_carres_q10=lb_r2, ljung_box_carres_p=p_r2,
    acf_rendements_lag1=float(acf(r.values, 1)[0]),
)
yr = d["Close"].resample("YE").last()
first = d["Close"].iloc[0]
annual = []
prev = first
for y, v in yr.items():
    ry = r[r.index.year == y.year]
    annual.append(dict(annee=y.year, debut=round(float(prev), 3), fin=round(float(v), 3),
                       rendement=float(v / prev - 1), volatilite=float(ry.std() * np.sqrt(365)),
                       n_jours=int(len(ry) + (1 if y.year == d.index[0].year else 0))))
    prev = v
summary["par_annee"] = annual
json.dump(summary, open(f"{OUT}/eda.json", "w"), ensure_ascii=False, indent=1)

# ---------- figures ----------
p = pd.read_csv("outputs/predictions.csv", parse_dates=["date_decision"])
bounds = {s: (p.loc[p.ensemble == s, "date_decision"].min(), p.loc[p.ensemble == s, "date_decision"].max())
          for s in ("train", "val", "test")}

fig, ax = plt.subplots(figsize=(8, 3.4))
ax.semilogy(d.index, d["Close"], color=INK, lw=1)
for s, c, lab in (("train", "#ececec", "Entraînement"), ("val", "#fbe0c3", "Validation"), ("test", "#cfe0f5", "Test")):
    ax.axvspan(*bounds[s], color=c, zorder=0, label=lab)
ax.set_ylabel("Prix de clôture (USD, échelle log)")
ax.set_title("DOT/USD quotidien, 2021-2026, et découpage chronologique", loc="left", fontsize=11)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.legend(loc="upper right", frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_prix.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 3.2))
bins = np.linspace(-0.3, 0.3, 121)
ax.hist(r, bins=bins, density=True, color=GREY, alpha=0.8, label="Rendements observés")
xs = np.linspace(-0.3, 0.3, 400)
ax.plot(xs, stats.norm.pdf(xs, r.mean(), r.std()), color=ACC, lw=1.6, label="Loi normale de même moyenne et écart-type")
ax.set_xlim(-0.25, 0.25)
ax.set_xlabel("Rendement logarithmique journalier")
ax.set_ylabel("Densité")
ax.set_title(f"Queues épaisses : kurtosis en excès de {summary['kurtosis_excess']:.1f}", loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_distribution.png"); plt.close(fig)

vol = r.rolling(30).std() * np.sqrt(365)
fig, ax = plt.subplots(figsize=(8, 3.0))
ax.plot(vol.index, vol * 100, color=INK, lw=1)
ax.axhline(summary["volatilite_annualisee"] * 100, color=ACC, lw=1, ls="--",
           label=f"Moyenne sur la période : {summary['volatilite_annualisee']*100:.0f} %")
ax.set_ylabel("Volatilité annualisée (%)")
ax.set_title("Volatilité glissante sur 30 jours : des régimes calmes et agités", loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_volatilite.png"); plt.close(fig)

lags = np.arange(1, 31)
ci = 1.96 / np.sqrt(len(r))
fig, axes = plt.subplots(1, 2, figsize=(8, 3.0), sharey=True)
for ax, series, title in ((axes[0], r.values, "Rendements"), (axes[1], (r ** 2).values, "Rendements au carré")):
    ax.bar(lags, acf(series, 30), color=ACC if title != "Rendements" else GREY, width=0.7)
    ax.axhspan(-ci, ci, color="#f1d9d9", zorder=0)
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_xlabel("Retard (jours)")
axes[0].set_ylabel("Autocorrélation")
fig.suptitle("Les rendements sont presque sans mémoire, leur amplitude ne l'est pas", x=0.01, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_acf.png"); plt.close(fig)

print(json.dumps({k: v for k, v in summary.items() if k != "par_annee"}, indent=1, ensure_ascii=False))
for a in annual:
    print(a)
