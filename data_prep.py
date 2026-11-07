"""Livrable 2 - Préparation des données DOT/USD (Investing.com, quotidien).

Étapes : lecture, nettoyage, variables explicatives, fenêtres temporelles,
découpage CHRONOLOGIQUE train / validation / test avec purge, normalisation
ajustée uniquement sur le train (aucune donnée future utilisée).
"""
import numpy as np
import pandas as pd

WINDOW = 20                     # longueur des fenêtres (jours) ; 4 si l'historique est très court
HORIZON = 5                     # cible : rendement log cumulé sur les 5 prochains jours
FEATURES = ["ret", "rng", "dvol", "madev20", "mom5", "mom20", "vol20", "rsi14"]
SPLIT = (0.60, 0.15, 0.25)      # train / validation / test (chronologique)


def parse_volume(s: str) -> float:
    s = str(s).strip().replace(",", "")
    if s in ("", "nan", "-"):
        return np.nan
    mult = {"K": 1e3, "M": 1e6, "B": 1e9}.get(s[-1], 1.0)
    return float(s[:-1]) * mult if s[-1] in "KMB" else float(s)


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df["Volume"] = df["Vol."].map(parse_volume)
    df = df.rename(columns={"Price": "Close"})
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df = df.sort_values("Date").drop_duplicates("Date").set_index("Date")
    # Jours manquants : on insère le jour avec O=H=L=C = clôture précédente (marché inchangé)
    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    missing = full.difference(df.index)
    df = df.reindex(full)
    df["Close"] = df["Close"].ffill()
    for c in ("Open", "High", "Low"):
        df[c] = df[c].fillna(df["Close"])
    df["Volume"] = df["Volume"].ffill().bfill()
    df = df.rename_axis("Date").reset_index()
    df.attrs["jours_ajoutes"] = [str(d.date()) for d in missing]
    assert df.isna().sum().sum() == 0
    return df


def rsi(close: pd.Series, n=14) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-diff.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Toutes les variables du jour t n'utilisent que des prix connus à la clôture de t."""
    d = df.copy()
    lc = np.log(d["Close"])
    d["ret"] = lc.diff()                                                    # rendement log du jour
    d["rng"] = (d["High"] - d["Low"]) / d["Close"]                          # amplitude intrajournalière
    d["dvol"] = np.log(d["Volume"] / d["Volume"].shift(1)).clip(-3, 3)      # variation du volume
    d["madev20"] = d["Close"] / d["Close"].rolling(20).mean() - 1           # écart à la MM20
    d["mom5"] = lc.diff(5)                                                  # momentum 5 j
    d["mom20"] = lc.diff(20)                                                # momentum 20 j
    d["vol20"] = d["ret"].rolling(20).std()                                 # volatilité 20 j
    d["rsi14"] = rsi(d["Close"]) / 100 - 0.5                                # RSI 14 recentré
    d["madev3"] = d["Close"] / d["Close"].rolling(3).mean() - 1             # pour la règle technique
    d["target"] = lc.shift(-HORIZON) - lc                                   # rendement log t -> t+H
    return d


def build_windows(d: pd.DataFrame):
    """Retourne X (n, WINDOW, nfeat), y (n,), idx (indice du dernier jour de la fenêtre)."""
    global WINDOW
    WINDOW = 20 if len(d) >= 300 else 4
    F = d[FEATURES].values
    T = d["target"].values
    X, y, idx = [], [], []
    for t in range(WINDOW - 1, len(d)):
        w = F[t - WINDOW + 1: t + 1]
        if np.isnan(w).any() or np.isnan(T[t]):
            continue
        X.append(w)
        y.append(T[t])
        idx.append(t)
    return np.array(X), np.array(y), np.array(idx)


def prepare(path: str):
    raw = load_raw(path)
    d = add_features(raw)
    X, y, idx = build_windows(d)
    n = len(y)
    n_tr = int(round(n * SPLIT[0]))
    n_va = int(round(n * SPLIT[1]))
    split = np.array(["train"] * n_tr + ["val"] * n_va + ["test"] * (n - n_tr - n_va), dtype=object)
    # Purge : les H dernières fenêtres de train et de val ont une cible qui chevauche l'ensemble suivant
    h = HORIZON if n > 100 else 0
    if h:
        split[n_tr - h: n_tr] = "purge"
        split[n_tr + n_va - h: n_tr + n_va] = "purge"
    split = split.astype(str)
    tr = split == "train"
    mu = X[tr].reshape(-1, X.shape[-1]).mean(0)
    sd = X[tr].reshape(-1, X.shape[-1]).std(0) + 1e-9
    y_sd = y[tr].std() + 1e-9
    Xs = (X - mu) / sd
    ys = y / y_sd
    d["split"] = ""
    d.loc[idx, "split"] = split
    return dict(raw=raw, df=d, X=Xs, y=ys, y_raw=y, idx=idx, split=split,
                scaler=dict(mu=mu, sd=sd, y_sd=y_sd))


if __name__ == "__main__":
    p = prepare("data/dot_usd_complet.csv")
    out = p["df"].copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    out.to_csv("outputs/base_donnees_preparee.csv", index=False)
    s = p["split"]
    print("jours:", len(p["raw"]), "| fenêtres:", len(s),
          "| train/val/test:", (s == "train").sum(), (s == "val").sum(), (s == "test").sum(),
          "| jours ajoutés:", p["raw"].attrs["jours_ajoutes"])
