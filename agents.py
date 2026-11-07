"""Agents autonomes + moteur de simulation financière (tout est virtuel).

Chaîne de décision à chaque jour t (information disponible à la clôture de t) :
  Agent 1 Prévision  -> rendement attendu de J+1 (moyenne et dispersion de l'ensemble GRU)
  Agent 2 Stratégie  -> signal ACHAT / VENTE / CONSERVATION
  Agent 3 Risque     -> valide, réduit ou bloque l'ordre (plafond d'exposition, stop-loss, coupe-circuit)
  Agent 4 Exécution  -> exécute à l'OUVERTURE de J+1 (frais + glissement), met à jour le portefeuille
                        et enregistre chaque événement dans la blockchain.
Aucune donnée postérieure à la clôture de t n'est utilisée pour décider.
"""
from dataclasses import dataclass, field
import numpy as np

CONFIG = dict(
    capital_initial=10_000.0,   # USD virtuels
    frais=0.0026,               # 0,26 % par ordre (ordre de grandeur des frais taker Kraken)
    glissement=0.0005,          # 0,05 % de glissement
    plafond_exposition=0.80,    # au plus 80 % de la valeur du portefeuille en DOT
    stop_loss=0.10,             # vente forcée si l'ouverture est 10 % sous le prix d'entrée
    coupe_circuit=0.85,         # plus d'achat si valeur < 85 % du capital initial
    taille_min=50.0,            # ordre minimum (USD)
    seuil_achat=0.006,          # rendement attendu minimal pour acheter (couvre les coûts aller-retour)
    seuil_vente=-0.002,         # rendement attendu en dessous duquel on sort
    confiance=0.5,              # achat seulement si mu > confiance * sigma_ensemble
)


@dataclass
class Portfolio:
    cash: float
    units: float = 0.0
    avg_cost: float = 0.0       # coût moyen par unité, frais d'achat inclus
    entry_price: float = 0.0    # prix d'exécution moyen (pour le stop-loss)

    def value(self, px):
        return self.cash + self.units * px


class ForecastAgent:
    agent_id = "AGENT_1_PREVISION"

    def __init__(self, models, X, y_sd, idx_to_row):
        self.models, self.X, self.y_sd, self.row = models, X, y_sd, idx_to_row

    def forecast(self, t):
        x = self.X[self.row[t]][None]
        preds = np.array([m.predict(x)[0] for m in self.models]) * self.y_sd
        return float(preds.mean()), float(preds.std())


class StrategyAgent:
    agent_id = "AGENT_2_STRATEGIE"

    def __init__(self, cfg):
        self.c = cfg

    def decide(self, mu, sigma, in_position):
        if mu > self.c["seuil_achat"] and mu > self.c["confiance"] * sigma:
            return "ACHAT"
        if in_position and mu < self.c["seuil_vente"]:
            return "VENTE"
        return "CONSERVATION"


class RiskAgent:
    agent_id = "AGENT_3_RISQUE"

    def __init__(self, cfg, cap=None, stop=None, circuit=None):
        self.c = cfg
        self.cap = cfg["plafond_exposition"] if cap is None else cap
        self.stop = cfg["stop_loss"] if stop is None else stop
        self.circuit = cfg["coupe_circuit"] if circuit is None else circuit

    def review(self, signal, pf: Portfolio, open_px, initial):
        eq = pf.value(open_px)
        if pf.units > 0 and self.stop and open_px <= pf.entry_price * (1 - self.stop):
            return "VENDRE", pf.units, "stop-loss déclenché"
        if pf.units > 0 and self.circuit and eq < self.circuit * initial:
            return "VENDRE", pf.units, "coupe-circuit : liquidation"
        if signal == "VENTE":
            return ("VENDRE", pf.units, "signal de vente validé") if pf.units > 0 else ("RIEN", 0.0, "aucune position à vendre")
        if signal == "ACHAT":
            if self.circuit and eq < self.circuit * initial:
                return "RIEN", 0.0, "coupe-circuit : achats interdits"
            fill = open_px * (1 + self.c["glissement"])
            room = self.cap * eq - pf.units * open_px
            budget = min(room, pf.cash / (1 + self.c["frais"]))
            if budget < self.c["taille_min"]:
                return "RIEN", 0.0, "plafond d'exposition atteint ou trésorerie insuffisante"
            return "ACHETER", budget / fill, f"achat validé (plafond {self.cap:.0%})"
        return "RIEN", 0.0, "conservation"


class ExecutionAgent:
    agent_id = "AGENT_4_EXECUTION"

    def __init__(self, cfg):
        self.c = cfg
        self.closed_pnls = []
        self.n_orders = 0

    def execute(self, action, qty, open_px, pf: Portfolio):
        if action == "RIEN" or qty <= 0:
            return dict(fill=None, qty=0.0, fee=0.0)
        self.n_orders += 1
        if action == "ACHETER":
            fill = open_px * (1 + self.c["glissement"])
            notional = qty * fill
            fee = notional * self.c["frais"]
            new_units = pf.units + qty
            pf.avg_cost = (pf.avg_cost * pf.units + notional + fee) / new_units
            pf.entry_price = (pf.entry_price * pf.units + fill * qty) / new_units
            pf.cash -= notional + fee
            pf.units = new_units
        else:
            fill = open_px * (1 - self.c["glissement"])
            notional = qty * fill
            fee = notional * self.c["frais"]
            self.closed_pnls.append(notional - fee - pf.avg_cost * qty)
            pf.cash += notional - fee
            pf.units -= qty
            if pf.units < 1e-12:
                pf.units, pf.avg_cost, pf.entry_price = 0.0, 0.0, 0.0
        return dict(fill=fill, qty=qty, fee=fee)


def simulate(df, decision_days, signal_fn, risk, cfg=CONFIG, ledger=None, name="agents"):
    """Boucle de simulation. df: DataFrame trié (Date, Open, Close...).
    decision_days: indices t ; exécution à l'ouverture de t+1, valorisation à la clôture de t+1.
    signal_fn(t, in_position) -> (signal, info)."""
    pf = Portfolio(cfg["capital_initial"])
    ex = ExecutionAgent(cfg)
    t0 = decision_days[0]
    dates, equity = [df["Date"].iloc[t0]], [pf.value(df["Close"].iloc[t0])]
    log = []
    for t in decision_days:
        d_dec, d_exe = df["Date"].iloc[t], df["Date"].iloc[t + 1]
        o, c = float(df["Open"].iloc[t + 1]), float(df["Close"].iloc[t + 1])
        signal, info = signal_fn(t, pf.units > 0)
        action, qty, reason = risk.review(signal, pf, o, cfg["capital_initial"])
        v_before = pf.value(o)
        res = ex.execute(action, qty, o, pf)
        v_after = pf.value(o)
        if ledger is not None:
            ds = d_dec.strftime("%Y-%m-%d")
            if "mu" in info:
                ledger.add_event("AGENT_1_PREVISION", ds, dict(
                    decision_sur_cloture=ds, horizon_jours=cfg.get("horizon_jours", 1), rendement_prevu_horizon=round(info["mu"], 6),
                    dispersion_ensemble=round(info["sigma"], 6)))
            ledger.add_event("AGENT_2_STRATEGIE", ds, dict(
                signal=signal, seuil_achat=cfg["seuil_achat"], seuil_vente=cfg["seuil_vente"]))
            ledger.add_event("AGENT_3_RISQUE", ds, dict(
                decision=action, quantite=round(qty, 6), raison=reason,
                plafond=risk.cap, stop_loss=risk.stop))
            ledger.add_event("AGENT_4_EXECUTION", d_exe.strftime("%Y-%m-%d"), dict(
                action_simulee=action, quantite=round(res["qty"], 6),
                prix_execution=None if res["fill"] is None else round(res["fill"], 6),
                frais=round(res["fee"], 4),
                valeur_portefeuille_avant=round(v_before, 4),
                valeur_portefeuille_apres=round(v_after, 4),
                tresorerie=round(pf.cash, 4), unites_DOT=round(pf.units, 6)))
        log.append(dict(date_decision=d_dec, date_execution=d_exe, signal=signal, action=action,
                        qty=qty, fill=res["fill"], fee=res["fee"], reason=reason,
                        mu=info.get("mu"), sigma=info.get("sigma"),
                        v_before=v_before, v_after=v_after))
        dates.append(d_exe)
        equity.append(pf.value(c))
    return dict(name=name, dates=dates, equity=np.array(equity), log=log, pf=pf,
                closed_pnls=ex.closed_pnls, n_orders=ex.n_orders)


def metrics(sim, cfg=CONFIG):
    eq = sim["equity"]
    v0, vT = eq[0], eq[-1]
    rets = eq[1:] / eq[:-1] - 1
    peak = np.maximum.accumulate(eq)
    mdd = float(((eq - peak) / peak).min())
    sd = rets.std(ddof=1) if len(rets) > 1 else np.nan
    sharpe = float(rets.mean() / sd * np.sqrt(365)) if sd and sd > 0 else float("nan")
    wins = [p for p in sim["closed_pnls"] if p > 0]
    n = len(eq) - 1
    return dict(
        strategie=sim["name"], valeur_initiale=float(v0), valeur_finale=float(vT),
        profit_net=float(vT - v0), rendement_total=float(vT / v0 - 1),
        rendement_annualise=float((vT / v0) ** (365 / n) - 1) if n > 0 else float("nan"),
        max_drawdown=mdd, sharpe=sharpe, n_ordres=int(sim["n_orders"]),
        n_trades_clotures=len(sim["closed_pnls"]),
        taux_reussite=float(len(wins) / len(sim["closed_pnls"])) if sim["closed_pnls"] else None,
        frais_totaux=float(sum(l["fee"] for l in sim["log"])), n_jours=int(n))
