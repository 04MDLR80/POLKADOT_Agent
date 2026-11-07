"""Simulation EN DIRECT des agents sur DOT/USD (paper trading : aucun ordre réel).

    python live_agent.py            # direct : prix réels Kraken, une décision par jour (clôture 00:00 UTC)
    python live_agent.py --rejeu    # rejoue la période de test en accéléré (1 jour toutes les 0,5 s)
    python live_agent.py --reset    # repart de 10 000 $ (efface l'état et le registre du direct)

Le tableau de bord s'ouvre sur http://localhost:8050 et se met à jour toutes les 3 secondes.
Arrêt : Ctrl+C. L'état est sauvegardé dans live/ et repris au prochain lancement.

Fonctionnement du direct :
  - toutes les 15 s : dernier prix DOT/USD (API publique Kraken, sans clé) -> valorisation du portefeuille ;
  - à chaque nouvelle bougie quotidienne clôturée : les 4 agents décident sur cette clôture,
    l'ordre est exécuté au prix courant (= ouverture du nouveau jour) et 4 blocs sont ajoutés au registre.
  - au premier lancement, les agents décident immédiatement sur la dernière bougie clôturée.
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.request
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents import ExecutionAgent, Portfolio, RiskAgent, StrategyAgent  # noqa: E402
from blockchain import Blockchain, verify_chain  # noqa: E402
from data_prep import FEATURES, add_features, load_raw  # noqa: E402
from gru_model import GRURegressor, LSTMRegressor  # noqa: E402

ICI = os.path.dirname(os.path.abspath(__file__))
DOSSIER = os.path.join(ICI, "live")
PORT = 8050
POLL_S = 15
KRAKEN = "https://api.kraken.com/0/public"
UA = {"User-Agent": "projet-deep-learning-simulation/1.0"}


# ----------------------------------------------------------------- modèle
class Modele:
    def __init__(self, chemin=os.path.join(ICI, "outputs", "modele_live.npz")):
        z = np.load(chemin, allow_pickle=False)
        self.meta = json.loads(str(z["meta"]))
        self.mu, self.sd, self.y_sd = z["mu"], z["sd"], float(z["y_sd"])
        cls = LSTMRegressor if self.meta["famille"] == "LSTM" else GRURegressor
        self.reseaux = []
        for j in range(self.meta["n_modeles"]):
            m = cls(len(FEATURES), self.meta["n_hid"])
            m.p = {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith(f"m{j}__")}
            self.reseaux.append(m)
        self.fenetre = self.meta["fenetre"]

    def prevoir(self, bougies: pd.DataFrame):
        """bougies : Date, Open, High, Low, Close, Volume (jours clôturés). Retourne (mu, sigma) du rendement à H jours."""
        f = add_features(bougies)[FEATURES].values[-self.fenetre:]
        if len(f) < self.fenetre or np.isnan(f).any():
            raise ValueError("historique insuffisant pour calculer les variables")
        x = ((f - self.mu) / self.sd)[None]
        p = np.array([m.predict(x)[0] for m in self.reseaux]) * self.y_sd
        return float(p.mean()), float(p.std())


# ----------------------------------------------------------------- source de prix
def _get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=15) as r:
        d = json.loads(r.read().decode())
    if d.get("error"):
        raise RuntimeError("Kraken : " + "; ".join(d["error"]))
    res = d["result"]
    return res[next(k for k in res if k != "last")]


def kraken_prix():
    return float(_get(f"{KRAKEN}/Ticker?pair=DOTUSD")["c"][0])


def kraken_bougies():
    """Bougies quotidiennes clôturées (la dernière renvoyée par Kraken est en cours : exclue)."""
    rows = _get(f"{KRAKEN}/OHLC?pair=DOTUSD&interval=1440")[:-1]
    df = pd.DataFrame([dict(Date=pd.Timestamp(int(r[0]), unit="s"), Open=float(r[1]), High=float(r[2]),
                            Low=float(r[3]), Close=float(r[4]), Volume=float(r[6])) for r in rows])
    df["Volume"] = df["Volume"].replace(0, np.nan).ffill().bfill()
    return df


# ----------------------------------------------------------------- simulateur (état + agents + registre)
class Simulateur:
    def __init__(self, modele, mode, reset=False):
        self.m, self.mode = modele, mode
        self.cfg = dict(modele.meta["config"])
        self.lock = threading.Lock()
        self.fichier = os.path.join(DOSSIER, f"etat_{mode}.json")
        self.fichier_reg = os.path.join(DOSSIER, f"registre_{mode}.json")
        os.makedirs(DOSSIER, exist_ok=True)
        self.statut = "démarrage"
        self.erreur = None
        if not reset and mode == "direct" and os.path.exists(self.fichier):
            self._charger()
        else:
            self._nouveau()

    # --- état
    def _nouveau(self):
        c = self.cfg["capital_initial"]
        self.pf = Portfolio(c)
        self.ex = ExecutionAgent(self.cfg)
        self.reg = Blockchain()
        self.reg.add_event("SYSTEME", "-", dict(
            mode=self.mode, actif="DOT/USD", source="Kraken (API publique)" if self.mode == "direct" else "historique Investing.com",
            modele=f"ensemble de {len(self.m.reseaux)} {self.m.meta['famille']} (fenêtre {self.m.fenetre} j, horizon {self.m.meta['horizon']} j)",
            configuration=self.cfg, avertissement="simulation : aucun ordre réel"))
        self.debut = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.derniere_bougie = None
        self.prix = None
        self.prix_depart = None
        self.bh_unites = None
        self.histo = []        # points de valorisation {t, prix, valeur, bh}
        self.decisions = []

    def _charger(self):
        e = json.load(open(self.fichier, encoding="utf-8"))
        self.pf = Portfolio(**e["portefeuille"])
        self.ex = ExecutionAgent(self.cfg)
        self.ex.closed_pnls, self.ex.n_orders = e["pnls"], e["n_ordres"]
        d = json.load(open(self.fichier_reg, encoding="utf-8"))
        ok, msg = verify_chain(d["chain"], d["difficulty"])
        if not ok:
            raise SystemExit(f"Registre falsifié ou abîmé ({msg}). Relancez avec --reset.")
        self.reg = Blockchain.__new__(Blockchain)
        self.reg.difficulty, self.reg.chain = d["difficulty"], d["chain"]
        for k in ("debut", "derniere_bougie", "prix", "prix_depart", "bh_unites", "histo", "decisions"):
            setattr(self, k, e[k])

    def sauver(self):
        if self.mode != "direct":
            return
        e = dict(portefeuille=self.pf.__dict__, pnls=self.ex.closed_pnls, n_ordres=self.ex.n_orders,
                 debut=self.debut, derniere_bougie=self.derniere_bougie, prix=self.prix, prix_depart=self.prix_depart,
                 bh_unites=self.bh_unites, histo=self.histo[-5000:], decisions=self.decisions)
        tmp = self.fichier + ".tmp"
        json.dump(e, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
        os.replace(tmp, self.fichier)
        self.reg.export(self.fichier_reg)

    # --- valorisation
    def maj_prix(self, prix, t=None):
        with self.lock:
            self.prix = prix
            if self.prix_depart is None:
                self.prix_depart = prix
                fill = prix * (1 + self.cfg["glissement"])
                self.bh_unites = self.cfg["capital_initial"] / (fill * (1 + self.cfg["frais"]))
            self.histo.append(dict(t=t or datetime.now(timezone.utc).isoformat(timespec="seconds"), prix=prix,
                                   valeur=round(self.pf.value(prix), 4), bh=round(self.bh_unites * prix, 4)))

    # --- décision des 4 agents
    def decider(self, bougies, prix_execution, quand=None):
        jour = bougies["Date"].iloc[-1].strftime("%Y-%m-%d")
        with self.lock:
            if self.derniere_bougie is not None and jour <= self.derniere_bougie:
                return None
            mu, sg = self.m.prevoir(bougies)
            signal = StrategyAgent(self.cfg).decide(mu, sg, self.pf.units > 0)
            risque = RiskAgent(self.cfg)
            action, qty, raison = risque.review(signal, self.pf, prix_execution, self.cfg["capital_initial"])
            v_av = self.pf.value(prix_execution)
            res = self.ex.execute(action, qty, prix_execution, self.pf)
            v_ap = self.pf.value(prix_execution)
            quand = quand or datetime.now(timezone.utc).isoformat(timespec="seconds")
            b1 = self.reg.add_event("AGENT_1_PREVISION", jour, dict(
                decision_sur_cloture=jour, cloture=float(bougies["Close"].iloc[-1]), horizon_jours=self.m.meta["horizon"],
                rendement_prevu_horizon=round(mu, 6), dispersion_ensemble=round(sg, 6)))
            self.reg.add_event("AGENT_2_STRATEGIE", jour, dict(
                signal=signal, seuil_achat=self.cfg["seuil_achat"], seuil_vente=self.cfg["seuil_vente"]))
            self.reg.add_event("AGENT_3_RISQUE", jour, dict(
                decision=action, quantite=round(qty, 6), raison=raison, plafond=risque.cap, stop_loss=risque.stop))
            b4 = self.reg.add_event("AGENT_4_EXECUTION", quand[:10], dict(
                action_simulee=action, quantite=round(res["qty"], 6),
                prix_execution=None if res["fill"] is None else round(res["fill"], 6), frais=round(res["fee"], 4),
                valeur_portefeuille_avant=round(v_av, 4), valeur_portefeuille_apres=round(v_ap, 4),
                tresorerie=round(self.pf.cash, 4), unites_DOT=round(self.pf.units, 6)))
            d = dict(jour=jour, execute_le=quand, cloture=float(bougies["Close"].iloc[-1]), mu=mu, sigma=sg,
                     signal=signal, action=action, quantite=res["qty"], prix=res["fill"], frais=res["fee"],
                     raison=raison, valeur_apres=v_ap, blocs=[b1["index"], b4["index"]], hash=b4["hash"])
            self.decisions.append(d)
            self.derniere_bougie = jour
            return d

    # --- vue pour le tableau de bord
    def vue(self):
        with self.lock:
            prix = self.prix
            c = self.cfg["capital_initial"]
            val = self.pf.value(prix) if prix else c
            bh = self.bh_unites * prix if prix and self.bh_unites else c
            gagnants = sum(1 for p in self.ex.closed_pnls if p > 0)
            hist = self.histo
            if len(hist) > 1500:
                pas = len(hist) // 1500 + 1
                hist = hist[::pas] + [hist[-1]]
            ok, msg = verify_chain(self.reg.chain, self.reg.difficulty)
            return dict(
                mode=self.mode, statut=self.statut, erreur=self.erreur, maintenant=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                debut=self.debut, prix=prix, prix_depart=self.prix_depart, capital=c, valeur=val, buy_hold=bh,
                tresorerie=self.pf.cash, unites=self.pf.units, prix_entree=self.pf.entry_price,
                exposition=(self.pf.units * prix / val) if prix and val else 0.0,
                coupe_circuit_actif=bool(self.cfg["coupe_circuit"] and val < self.cfg["coupe_circuit"] * c),
                n_ordres=self.ex.n_orders, trades_clotures=len(self.ex.closed_pnls), trades_gagnants=gagnants,
                derniere_bougie=self.derniere_bougie, decisions=self.decisions[-60:], n_decisions=len(self.decisions),
                histo=hist, config=self.cfg, modele=self.m.meta["famille"], n_reseaux=len(self.m.reseaux),
                horizon=self.m.meta["horizon"], blocs=self.reg.chain[-8:], n_blocs=len(self.reg.chain),
                chaine_valide=ok, chaine_message=msg)


# ----------------------------------------------------------------- boucles
def boucle_direct(sim, stop):
    bougies, maj_bougies = None, 0.0
    while not stop.is_set():
        try:
            prix = kraken_prix()
            sim.maj_prix(prix)
            hier = (datetime.now(timezone.utc) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            en_retard = bougies is not None and bougies["Date"].iloc[-1].strftime("%Y-%m-%d") < hier
            if bougies is None or en_retard or time.time() - maj_bougies > 300:
                bougies, maj_bougies = kraken_bougies(), time.time()
            d = sim.decider(bougies, prix)
            if d:
                print(f"[{d['execute_le']}] décision sur la clôture du {d['jour']} : prévu {d['mu']*100:+.2f} %, "
                      f"signal {d['signal']}, {d['action']} ({d['raison']}) -> valeur {d['valeur_apres']:.2f} $", flush=True)
            sim.statut, sim.erreur = "en direct", None
            sim.sauver()
        except Exception as e:  # réseau coupé, API indisponible... on réessaie au tour suivant
            sim.statut, sim.erreur = "erreur de connexion, nouvel essai dans 15 s", str(e)[:200]
            print("Erreur :", e, flush=True)
        stop.wait(POLL_S)


def boucle_rejeu(sim, stop, vitesse):
    """Rejoue la période de test : décision sur la clôture de J, exécution à l'ouverture de J+1, valorisation à la clôture."""
    raw = load_raw(os.path.join(ICI, "data", "dot_usd_complet.csv"))
    p = pd.read_csv(os.path.join(ICI, "outputs", "predictions.csv"))
    debut_test = p.loc[p.ensemble == "test", "date_decision"].min()
    i0 = int(raw.index[raw["Date"] == pd.Timestamp(debut_test)][0])
    sim.statut = "rejeu en cours"
    sim.maj_prix(float(raw["Close"].iloc[i0]), raw["Date"].iloc[i0].strftime("%Y-%m-%d"))
    for t in range(i0, len(raw) - 1):
        if stop.is_set():
            return
        jour_suiv = raw["Date"].iloc[t + 1].strftime("%Y-%m-%d")
        sim.decider(raw.iloc[: t + 1], float(raw["Open"].iloc[t + 1]), quand=jour_suiv)
        sim.maj_prix(float(raw["Close"].iloc[t + 1]), jour_suiv)
        stop.wait(vitesse)
    sim.statut = "rejeu terminé"


# ----------------------------------------------------------------- serveur du tableau de bord
def serveur(sim):
    page = open(os.path.join(ICI, "live_dashboard.html"), encoding="utf-8").read()

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/api/etat"):
                corps, typ = json.dumps(sim.vue(), ensure_ascii=False).encode(), "application/json; charset=utf-8"
            elif self.path in ("/", "/index.html"):
                corps, typ = page.encode(), "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", typ)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(corps)

        def log_message(self, *a):
            pass

    return ThreadingHTTPServer(("127.0.0.1", PORT), H)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rejeu", action="store_true", help="rejouer la période de test en accéléré")
    ap.add_argument("--vitesse", type=float, default=0.5, help="secondes par jour en mode rejeu (défaut 0,5)")
    ap.add_argument("--reset", action="store_true", help="repartir de 10 000 $ en mode direct")
    ap.add_argument("--sans-navigateur", action="store_true", help="ne pas ouvrir le navigateur")
    a = ap.parse_args()

    sim = Simulateur(Modele(), "rejeu" if a.rejeu else "direct", reset=a.reset)
    stop = threading.Event()
    srv = serveur(sim)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    cible = boucle_rejeu if a.rejeu else boucle_direct
    args = (sim, stop, a.vitesse) if a.rejeu else (sim, stop)
    th = threading.Thread(target=cible, args=args, daemon=True)
    th.start()
    url = f"http://localhost:{PORT}"
    print(f"Simulation {'REJEU' if a.rejeu else 'EN DIRECT'} — tableau de bord : {url}  (Ctrl+C pour arrêter)", flush=True)
    if not a.sans_navigateur:
        webbrowser.open(url)
    try:
        while th.is_alive():
            th.join(0.5)
        print("Rejeu terminé. Le tableau de bord reste ouvert ; Ctrl+C pour quitter.", flush=True)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop.set()
        sim.sauver()
        srv.shutdown()
        print("\nArrêt. État sauvegardé dans le dossier live/.")


if __name__ == "__main__":
    main()
