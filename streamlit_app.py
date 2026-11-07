"""Tableau de bord Streamlit — Agents autonomes sur DOT/USD (simulation, aucun ordre réel).

    pip install -r requirements.txt
    streamlit run streamlit_app.py

Onglets : Direct (prix Kraken en temps réel) · Résultats du test · Prévisions · Transactions ·
Blockchain · Données & modèle.  Le direct tourne dans un thread en arrière-plan ; l'état est
sauvegardé dans live/ et repris au redémarrage.
"""
import json
import os
import sys
import threading

import pandas as pd
import streamlit as st

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
OUT = os.path.join(ICI, "outputs")

st.set_page_config(page_title="Agents DOT/USD", page_icon="📈", layout="wide")


# ------------------------------------------------------------------ données statiques
@st.cache_data
def charger():
    res = json.load(open(os.path.join(OUT, "results.json"), encoding="utf-8"))
    met = json.load(open(os.path.join(OUT, "metrics.json"), encoding="utf-8"))
    pred = pd.read_csv(os.path.join(OUT, "predictions.csv"), parse_dates=["date_decision"])
    tr = pd.read_csv(os.path.join(OUT, "transactions.csv"))
    led = json.load(open(os.path.join(OUT, "ledger.json"), encoding="utf-8"))
    return res, met, pred, tr, led


def pct(x, d=1):
    return "–" if x is None or pd.isna(x) else f"{x * 100:+.{d}f} %"


def usd(x):
    return "–" if x is None or pd.isna(x) else f"{x:,.2f} $".replace(",", " ")


res, met, pred, tr, led = charger()
mod = met["modele_agents"]

# ------------------------------------------------------------------ en-tête
st.title("📈 Agents autonomes — DOT/USD")
st.caption(f"Prévision {mod} (ensemble de 5 réseaux) → agents Stratégie, Risque, Exécution → "
           "portefeuille virtuel → registre blockchain. **Simulation : aucun ordre réel.**")

with st.sidebar:
    st.header("Projet Deep Learning GSTR S3")
    d = met["donnees"]
    st.write(f"**Données** : {d['n_jours']} jours, {d['debut']} → {d['fin']}")
    st.write(f"**Découpage** : train {d['n_train']} · validation {d['n_val']} · test {d['n_test']}")
    st.write(f"**Fenêtre** : {d['fenetre']} j · **horizon** : {d['horizon']} j")
    st.write(f"**Modèle retenu** : {mod}, {met['retenu']['n_hid']} unités")
    st.divider()
    st.caption("Les résultats ne montrent pas de pouvoir prédictif robuste (voir l'onglet Prévisions).")

t_live, t_res, t_prev, t_tx, t_bc, t_data = st.tabs(
    ["🔴 Direct", "📊 Résultats du test", "🔮 Prévisions", "🧾 Transactions", "⛓️ Blockchain", "🗂️ Données & modèle"])

# ------------------------------------------------------------------ DIRECT
with t_live:
    mode = st.radio("Mode", ["direct", "rejeu"], horizontal=True,
                    format_func=lambda m: "Direct (prix réels Kraken)" if m == "direct" else "Rejeu accéléré du test")
    vitesse = st.slider("Rejeu : secondes par jour", 0.1, 3.0, 0.5, 0.1) if mode == "rejeu" else 0.5

    @st.cache_resource
    def demarrer(mode, vitesse):
        import live_agent as la
        sim = la.Simulateur(la.Modele(), mode)
        stop = threading.Event()
        if mode == "rejeu":
            th = threading.Thread(target=la.boucle_rejeu, args=(sim, stop, vitesse), daemon=True)
        else:
            th = threading.Thread(target=la.boucle_direct, args=(sim, stop), daemon=True)
        th.start()
        return sim

    sim = demarrer(mode, vitesse)
    if mode == "direct":
        if st.button("Repartir de 10 000 $"):
            import live_agent as la
            for f in (sim.fichier, sim.fichier_reg):
                if os.path.exists(f):
                    os.remove(f)
            sim._nouveau()
            sim.derniere_bougie = None
            st.rerun()

    @st.fragment(run_every=5)
    def affichage():
        v = sim.vue()
        if v["erreur"]:
            st.warning(f"{v['statut']} — {v['erreur']}")
        else:
            st.caption(f"Statut : {v['statut']} · mis à jour {v['maintenant']} UTC")
        cap = v["capital"]
        c = st.columns(5)
        c[0].metric("Prix DOT/USD", "–" if not v["prix"] else f"{v['prix']:.4f} $")
        c[1].metric("Portefeuille des agents", usd(v["valeur"]), pct(v["valeur"] / cap - 1, 2))
        c[2].metric("Buy and Hold", usd(v["buy_hold"]), pct(v["buy_hold"] / cap - 1, 2))
        c[3].metric("Exposition", f"{v['exposition'] * 100:.0f} %", f"{v['unites']:.3f} DOT")
        c[4].metric("Ordres / trades clôturés", f"{v['n_ordres']} / {v['trades_clotures']}")
        if v["coupe_circuit_actif"]:
            st.error("Coupe-circuit actif : les achats sont bloqués (valeur < 85 % du capital).")
        h = pd.DataFrame(v["histo"])
        if len(h) > 1:
            h["t"] = pd.to_datetime(h["t"], errors="coerce")
            st.line_chart(h.set_index("t")[["valeur", "bh"]].rename(
                columns={"valeur": "Agents", "bh": "Buy and Hold"}), height=320)
            st.line_chart(h.set_index("t")[["prix"]].rename(columns={"prix": "DOT/USD"}), height=200)
        else:
            st.info("En attente des premiers prix…")
        st.subheader("Dernières décisions")
        if v["decisions"]:
            dd = pd.DataFrame(v["decisions"][::-1][:15])
            st.dataframe(dd, use_container_width=True, hide_index=True)
        else:
            st.write("Aucune décision pour l'instant (une par jour, à la clôture de 00:00 UTC).")
        ok = v["chaine_valide"]
        (st.success if ok else st.error)(f"Blockchain : {v['chaine_message']} ({v['n_blocs']} blocs)")
    affichage()

# ------------------------------------------------------------------ RÉSULTATS
with t_res:
    st.subheader("Performance sur la période de test (05/06/2025 → 31/10/2026)")
    s = pd.DataFrame(met["strategies"] + met["ablation"])
    k = st.columns(4)
    for col, nom in zip(k, ["Agents LSTM", "Règle technique (MM20)", "Buy and Hold"]):
        r = s[s.strategie == nom].iloc[0]
        col.metric(nom, usd(r.valeur_finale), pct(r.rendement_total))
    r = s[s.strategie == "Agents LSTM sans coupe-circuit"].iloc[0]
    k[3].metric("LSTM sans coupe-circuit", usd(r.valeur_finale), pct(r.rendement_total))

    cur = {**res["courbes"], **res["courbes_supp"]}
    choix = st.multiselect("Courbes", list(cur), default=list(res["courbes"]))
    if choix:
        df = pd.DataFrame({n: pd.Series(cur[n]["valeur"], index=pd.to_datetime(cur[n]["dates"])) for n in choix})
        st.line_chart(df, height=380)
    aff = s.rename(columns={
        "strategie": "Stratégie", "valeur_finale": "Valeur finale ($)", "rendement_total": "Rendement",
        "max_drawdown": "Drawdown max", "sharpe": "Sharpe", "n_ordres": "Ordres", "frais_totaux": "Frais ($)",
        "taux_reussite": "Taux de réussite"})[["Stratégie", "Valeur finale ($)", "Rendement", "Drawdown max",
                                                "Sharpe", "Ordres", "Frais ($)", "Taux de réussite"]]
    st.dataframe(aff, use_container_width=True, hide_index=True)
    st.info("L'écart avec le Buy and Hold vient surtout du coupe-circuit (achats bloqués pendant la chute), "
            "pas d'une prévision supérieure : sans lui, les agents perdent 36,9 %.")
    st.subheader("Sensibilité aux paramètres de risque et aux frais")
    st.dataframe(pd.DataFrame(met["sensibilite"]), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ PRÉVISIONS
with t_prev:
    st.subheader("Erreur de prévision (rendement à 5 jours)")
    rows = []
    for ens, d_ in met["prevision"].items():
        for mdl, m in d_.items():
            rows.append(dict(ensemble=ens, modèle=mdl, MAE=m["MAE"], RMSE=m["RMSE"], direction=m["direction"]))
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Sur le test, ni le LSTM ni le GRU ne battent la prévision naïve « moyenne du train » (RMSE).")
    ens = st.selectbox("Ensemble", ["test", "validation", "train"])
    p = pred[pred.ensemble == ens].set_index("date_decision")
    cols = {"rendement_reel": "Réel", "prevu_LSTM": "LSTM", "prevu_GRU": "GRU"}
    st.line_chart(p[list(cols)].rename(columns=cols), height=340)
    st.subheader("Courbes d'apprentissage (5 réseaux)")
    i = st.slider("Réseau", 1, len(res["apprentissage"]), 1) - 1
    a = res["apprentissage"][i]
    st.line_chart(pd.DataFrame({"train": a["train"], "validation": a["val"]}), height=260)

# ------------------------------------------------------------------ TRANSACTIONS
with t_tx:
    st.subheader("Décisions et ordres des agents (test)")
    f = st.multiselect("Action", sorted(tr.action.unique()), default=[a for a in tr.action.unique() if a != "RIEN"])
    st.dataframe(tr[tr.action.isin(f)] if f else tr, use_container_width=True, hide_index=True)
    st.line_chart(tr.assign(date=pd.to_datetime(tr.date_decision)).set_index("date")[["v_after"]]
                  .rename(columns={"v_after": "Valeur du portefeuille"}), height=260)

# ------------------------------------------------------------------ BLOCKCHAIN
with t_bc:
    from blockchain import verify_chain
    chain, diff = led["chain"], led["difficulty"]
    ok, msg = verify_chain(chain, diff)
    (st.success if ok else st.error)(f"{msg} — difficulté {diff} (hash commençant par {'0' * diff})")
    ag = pd.Series([b["agent"] for b in chain]).value_counts()
    st.bar_chart(ag)
    n = st.slider("Derniers blocs affichés", 5, 100, 20)
    st.dataframe(pd.DataFrame([{k: (json.dumps(b[k], ensure_ascii=False) if k == "data" else b[k])
                                for k in ("index", "sim_time", "agent", "data", "prev_hash", "nonce", "hash")}
                               for b in chain[-n:]][::-1]), use_container_width=True, hide_index=True)
    st.download_button("Télécharger ledger.json (à partager avec le professeur)",
                       json.dumps(led, ensure_ascii=False), "ledger.json", "application/json")

# ------------------------------------------------------------------ DONNÉES & MODÈLE
with t_data:
    px = pd.Series(res["prix"]["close"], index=pd.to_datetime(res["prix"]["dates"]), name="Clôture")
    st.subheader("Prix de clôture DOT/USD")
    st.line_chart(px, height=300)
    st.subheader("Recherche d'hyperparamètres (RMSE de validation)")
    hp = [dict(famille=f, **r) for f, L in met["recherche_hyperparametres"].items() for r in L]
    st.dataframe(pd.DataFrame(hp), use_container_width=True, hide_index=True)
    st.write("Vérification du gradient (différences finies) :", met["verification_gradient"])
    st.subheader("Configuration des agents")
    st.json(met["configuration"])
    eda = os.path.join(OUT, "eda")
    if os.path.isdir(eda):
        st.subheader("Analyse exploratoire")
        imgs = sorted(f for f in os.listdir(eda) if f.endswith(".png"))
        for c, f in zip(st.columns(2) * 2, imgs):
            c.image(os.path.join(eda, f), caption=f)
