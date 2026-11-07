"""GRU et LSTM implémentés en NumPy (rétropropagation à travers le temps écrite à la main).

PyTorch/TensorFlow ne sont pas installables dans cet environnement ; les deux
réseaux sont donc codés directement. Les gradients sont vérifiés par
différences finies (voir check_gradient). Sortie : rendement à l'horizon,
normalisé (régression, perte MSE + L2).
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class GRURegressor:
    def __init__(self, n_in, n_hid=8, seed=0):
        r = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(n_hid)
        self.p = {}
        for g in "zrn":
            self.p["W" + g] = r.uniform(-s, s, (n_in, n_hid))
            self.p["U" + g] = r.uniform(-s, s, (n_hid, n_hid))
            self.p["b" + g] = np.zeros(n_hid)
        self.p["Wo"] = r.uniform(-s, s, (n_hid, 1))
        self.p["bo"] = np.zeros(1)
        self.H = n_hid

    # ---------- passe avant ----------
    def forward(self, X):
        P, (B, L, _) = self.p, X.shape
        h = np.zeros((B, self.H))
        cache = []
        for t in range(L):
            x = X[:, t, :]
            z = sigmoid(x @ P["Wz"] + h @ P["Uz"] + P["bz"])
            r = sigmoid(x @ P["Wr"] + h @ P["Ur"] + P["br"])
            n = np.tanh(x @ P["Wn"] + (r * h) @ P["Un"] + P["bn"])
            hn = (1 - z) * n + z * h
            cache.append((x, h, z, r, n))
            h = hn
        y = (h @ P["Wo"] + P["bo"]).ravel()
        return y, (cache, h)

    def predict(self, X):
        return self.forward(X)[0]

    # ---------- perte + gradients ----------
    def loss_grad(self, X, t, l2=0.0):
        P = self.p
        y, (cache, hL) = self.forward(X)
        B = len(t)
        err = y - t
        loss = np.mean(err ** 2) + l2 * sum((P[k] ** 2).sum() for k in P if k[0] in "WU")
        G = {k: np.zeros_like(v) for k, v in P.items()}
        dy = (2.0 / B) * err[:, None]
        G["Wo"] = hL.T @ dy
        G["bo"] = dy.sum(0)
        dh = dy @ P["Wo"].T
        for (x, hp, z, r, n) in reversed(cache):
            dn = dh * (1 - z)
            dz = dh * (hp - n)
            dhp = dh * z
            dan = dn * (1 - n ** 2)
            G["Wn"] += x.T @ dan
            G["Un"] += (r * hp).T @ dan
            G["bn"] += dan.sum(0)
            drh = dan @ P["Un"].T
            dr = drh * hp
            dhp += drh * r
            daz = dz * z * (1 - z)
            dar = dr * r * (1 - r)
            G["Wz"] += x.T @ daz
            G["Uz"] += hp.T @ daz
            G["bz"] += daz.sum(0)
            G["Wr"] += x.T @ dar
            G["Ur"] += hp.T @ dar
            G["br"] += dar.sum(0)
            dhp += daz @ P["Uz"].T + dar @ P["Ur"].T
            dh = dhp
        if l2:
            for k in P:
                if k[0] in "WU":
                    G[k] += 2 * l2 * P[k]
        return loss, G

    # ---------- entraînement (Adam + arrêt anticipé sur la validation) ----------
    def fit(self, Xtr, ytr, Xva, yva, epochs=600, lr=0.01, l2=1e-3, patience=80, clip=1.0):
        P = self.p
        m = {k: np.zeros_like(v) for k, v in P.items()}
        v = {k: np.zeros_like(x) for k, x in P.items()}
        best, best_p, bad, hist = np.inf, None, 0, []
        for ep in range(1, epochs + 1):
            loss, G = self.loss_grad(Xtr, ytr, l2)
            gn = np.sqrt(sum((g ** 2).sum() for g in G.values()))
            sc = min(1.0, clip / (gn + 1e-12))
            for k in P:
                g = G[k] * sc
                m[k] = 0.9 * m[k] + 0.1 * g
                v[k] = 0.999 * v[k] + 0.001 * g * g
                P[k] -= lr * (m[k] / (1 - 0.9 ** ep)) / (np.sqrt(v[k] / (1 - 0.999 ** ep)) + 1e-8)
            vl = float(np.mean((self.predict(Xva) - yva) ** 2))
            hist.append((loss, vl))
            if vl < best - 1e-6:
                best, best_p, bad = vl, {k: x.copy() for k, x in P.items()}, 0
            else:
                bad += 1
                if bad >= patience:
                    break
        self.p = best_p
        return hist, best


class LSTMRegressor(GRURegressor):
    """LSTM une couche : portes d'entrée i, d'oubli f, de sortie o, candidat g, cellule c.
    Réutilise fit() et predict() du GRU ; seules la passe avant et le gradient changent."""

    def __init__(self, n_in, n_hid=8, seed=0):
        r = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(n_hid)
        self.p = {}
        for g in "ifog":
            self.p["W" + g] = r.uniform(-s, s, (n_in, n_hid))
            self.p["U" + g] = r.uniform(-s, s, (n_hid, n_hid))
            self.p["b" + g] = np.ones(n_hid) if g == "f" else np.zeros(n_hid)  # biais d'oubli à 1
        self.p["Wo_"] = r.uniform(-s, s, (n_hid, 1))
        self.p["bo_"] = np.zeros(1)
        self.H = n_hid

    def forward(self, X):
        P, (B, L, _) = self.p, X.shape
        h = np.zeros((B, self.H))
        c = np.zeros((B, self.H))
        cache = []
        for t in range(L):
            x = X[:, t, :]
            i = sigmoid(x @ P["Wi"] + h @ P["Ui"] + P["bi"])
            f = sigmoid(x @ P["Wf"] + h @ P["Uf"] + P["bf"])
            o = sigmoid(x @ P["Wo"] + h @ P["Uo"] + P["bo"])
            g = np.tanh(x @ P["Wg"] + h @ P["Ug"] + P["bg"])
            cn = f * c + i * g
            tc = np.tanh(cn)
            cache.append((x, h, c, i, f, o, g, tc))
            h, c = o * tc, cn
        y = (h @ P["Wo_"] + P["bo_"]).ravel()
        return y, (cache, h)

    def loss_grad(self, X, t, l2=0.0):
        P = self.p
        y, (cache, hL) = self.forward(X)
        B = len(t)
        err = y - t
        loss = np.mean(err ** 2) + l2 * sum((P[k] ** 2).sum() for k in P if k[0] in "WU" and k != "Wo_")
        G = {k: np.zeros_like(v) for k, v in P.items()}
        dy = (2.0 / B) * err[:, None]
        G["Wo_"] = hL.T @ dy
        G["bo_"] = dy.sum(0)
        dh = dy @ P["Wo_"].T
        dc = np.zeros_like(dh)
        for (x, hp, cp, i, f, o, g, tc) in reversed(cache):
            do = dh * tc
            dc = dc + dh * o * (1 - tc ** 2)
            di, df, dg = dc * g, dc * cp, dc * i
            dcp = dc * f
            pre = {"i": di * i * (1 - i), "f": df * f * (1 - f), "o": do * o * (1 - o), "g": dg * (1 - g ** 2)}
            dhp = np.zeros_like(hp)
            for k, a in pre.items():
                G["W" + k] += x.T @ a
                G["U" + k] += hp.T @ a
                G["b" + k] += a.sum(0)
                dhp += a @ P["U" + k].T
            dh, dc = dhp, dcp
        if l2:
            for k in P:
                if k[0] in "WU" and k != "Wo_":
                    G[k] += 2 * l2 * P[k]
        return loss, G


def check_gradient(seed=1, cls=None):
    cls = cls or GRURegressor
    r = np.random.default_rng(seed)
    X, t = r.normal(size=(5, 4, 4)), r.normal(size=5)
    m = cls(4, 6, seed)
    _, G = m.loss_grad(X, t)
    worst = 0.0
    for k in m.p:
        flat = m.p[k].ravel()
        for i in r.choice(len(flat), size=min(4, len(flat)), replace=False):
            old = flat[i]
            flat[i] = old + 1e-6
            lp, _ = m.loss_grad(X, t)
            flat[i] = old - 1e-6
            lm, _ = m.loss_grad(X, t)
            flat[i] = old
            num = (lp - lm) / 2e-6
            ana = G[k].ravel()[i]
            worst = max(worst, abs(num - ana) / (abs(num) + abs(ana) + 1e-12))
    return worst


if __name__ == "__main__":
    print("GRU  - erreur relative max du gradient :", check_gradient())
    print("LSTM - erreur relative max du gradient :", check_gradient(cls=LSTMRegressor))
