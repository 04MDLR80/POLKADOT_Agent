"""Tableau de bord — agents autonomes sur DOT/USD (simulation, aucun ordre réel).

    pip install -r requirements.txt
    streamlit run streamlit_app.py

Direct : lit l'état que GitHub Actions écrit dans live/ (trade_daily.py).  Rejeu : rejoue le test en accéléré.
"""
import json
import os
import sys
import threading
from datetime import datetime, timedelta, timezone

import altair as alt
import pandas as pd
import streamlit as st

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
OUT = os.path.join(ICI, "outputs")

# ------------------------------------------------------------------ design
INK, SLATE, MUTED, LINE, SURF = "#141821", "#5A6372", "#8A93A1", "#E3E6EB", "#FFFFFF"
POS, NEG, ACCENT = "#0B8A5A", "#C2362B", "#E6007A"
SERIES = {  # palette validée (écart CVD >= 10, contraste >= 3:1) ; une couleur = une entité, partout
    "Agents": ACCENT, "Agents LSTM": ACCENT, "Buy and Hold": "#2F6FB5", "Règle technique (MM20)": "#B87800",
    "Agents GRU": "#7A5CCF", "Agents LSTM sans coupe-circuit": "#00907F",
    "Réel": "#2F6FB5", "LSTM": ACCENT, "GRU": "#7A5CCF", "train": "#2F6FB5", "validation": ACCENT,
}

st.set_page_config(page_title="Agents DOT/USD", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp, button, input {{ font-family: 'Hanken Grotesk', system-ui, sans-serif !important; }}
.stApp {{ background: #F6F7F9; }}
[data-testid="stHeader"] {{ background: transparent; }}
footer, #MainMenu {{ visibility: hidden; }}
.block-container {{ max-width: 1240px; padding: 1.6rem 1.5rem 4rem; }}
h1,h2,h3 {{ letter-spacing: -0.02em; color: {INK}; }}
h3 {{ font-size: 1.05rem !important; font-weight: 650 !important; margin: 1.6rem 0 .5rem !important; }}
.num {{ font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }}

/* en-tête */
.top {{ display:flex; justify-content:space-between; align-items:flex-end; gap:1rem; flex-wrap:wrap; padding-bottom:1.1rem; }}
.top h1 {{ font-size:1.65rem; font-weight:800; margin:0; line-height:1.15; padding:0; }}
.top p {{ margin:.3rem 0 0; color:{SLATE}; font-size:.92rem; max-width:62ch; }}
.pill {{ display:inline-flex; align-items:center; gap:.5rem; background:{SURF}; border:1px solid {LINE}; border-radius:999px;
        padding:.38rem .85rem; font-size:.82rem; color:{SLATE}; white-space:nowrap; }}
.pill b {{ color:{INK}; font-weight:650; }}
.dot {{ width:8px; height:8px; border-radius:50%; background:{POS}; box-shadow:0 0 0 3px rgba(11,138,90,.18); }}
.dot.warn {{ background:#D98E04; box-shadow:0 0 0 3px rgba(217,142,4,.2); }}
.dot.off {{ background:{MUTED}; box-shadow:none; }}

/* bandeau d'indicateurs : un seul bloc, séparé par des filets */
.kpis {{ display:grid; grid-template-columns: 1.5fr repeat(4, 1fr); background:{SURF}; border:1px solid {LINE};
         border-radius:12px; overflow:hidden; }}
.kpi {{ padding:1rem 1.2rem 1.05rem; border-left:1px solid {LINE}; min-width:0; }}
.kpi:first-child {{ border-left:none; }}
.kpi .l {{ font-size:.8rem; color:{SLATE}; font-weight:550; }}
.kpi .v {{ font-size:1.55rem; font-weight:750; color:{INK}; margin-top:.2rem; line-height:1.15; }}
.kpi.main .v {{ font-size:2.2rem; font-weight:800; }}
.kpi .d {{ font-size:.82rem; margin-top:.25rem; font-weight:600; color:{SLATE}; }}
.kpi .d.pos {{ color:{POS}; }} .kpi .d.neg {{ color:{NEG}; }}
@media (max-width: 900px) {{ .kpis {{ grid-template-columns: 1fr 1fr; }} .kpi {{ border-top:1px solid {LINE}; }} .kpi:nth-child(odd) {{ border-left:none; }} }}

/* panneaux */
[data-testid="stVerticalBlockBorderWrapper"] {{ background:{SURF}; border-color:{LINE} !important; border-radius:12px !important; }}
.card-t {{ font-size:.82rem; color:{SLATE}; font-weight:600; margin-bottom:.55rem; }}
.badge {{ display:inline-block; padding:.2rem .65rem; border-radius:6px; font-weight:700; font-size:.82rem; }}
.badge.ACHAT {{ background:#E3F4EC; color:{POS}; }} .badge.VENTE {{ background:#FBE7E5; color:{NEG}; }}
.badge.CONSERVATION {{ background:#EDEFF3; color:{SLATE}; }}
.row {{ display:flex; justify-content:space-between; gap:1rem; padding:.45rem 0; border-top:1px solid {LINE}; font-size:.88rem; }}
.row span:first-child {{ color:{SLATE}; }} .row span:last-child {{ font-weight:600; text-align:right; }}
.gauge {{ position:relative; height:10px; border-radius:6px; margin:1.35rem 0 .35rem; background:#EDEFF3; overflow:visible; }}
.gauge .z {{ position:absolute; top:0; bottom:0; }}
.gauge .m {{ position:absolute; top:-5px; width:3px; height:20px; background:{INK}; border-radius:2px; transform:translateX(-50%); }}
.gauge-l {{ display:flex; justify-content:space-between; font-size:.74rem; color:{MUTED}; }}

/* onglets */
button[data-baseweb="tab"] {{ font-weight:600; font-size:.95rem; padding:.7rem 1rem; }}
[data-baseweb="tab-highlight"] {{ background:{ACCENT} !important; height:3px !important; }}
[data-baseweb="tab-border"] {{ background:{LINE} !important; }}
[data-testid="stDataFrame"] {{ border:1px solid {LINE}; border-radius:10px; overflow:hidden; }}
.note {{ border-left:3px solid {ACCENT}; background:{SURF}; padding:.7rem 1rem; color:{SLATE}; font-size:.9rem; border-radius:0 8px 8px 0; margin:.8rem 0; }}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ utilitaires
def fr(x, d=2):
    return "–" if x is None or pd.isna(x) else f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


def usd(x, d=2):
    return fr(x, d) + " $" if x is not None and not pd.isna(x) else "–"


def pct(x, d=1, signe=True):
    if x is None or pd.isna(x):
        return "–"
    return (f"{x * 100:+.{d}f}" if signe else f"{x * 100:.{d}f}").replace(".", ",").replace("-", "−") + " %"


def tone(x):
    return "pos" if x > 0 else "neg" if x < 0 else ""


def kpi(label, valeur, sous="", t="", main=False):
    return (f'<div class="kpi{" main" if main else ""}"><div class="l">{label}</div><div class="v num">{valeur}</div>'
            f'<div class="d {t} num">{sous}</div></div>')


def courbes(df, hauteur=300, fmt=",.0f", zero=False, etiquette="valeur"):
    """Courbes multi-séries : 2 px, grille discrète, curseur + infobulle, légende en haut."""
    df = df.copy()
    df.index.name = "t"
    noms = list(df.columns)
    wide = df.rename(columns={n: f"s{i}" for i, n in enumerate(noms)}).reset_index()
    long = df.reset_index().melt("t", var_name="série", value_name="v").dropna()
    couleurs = [SERIES.get(n, MUTED) for n in noms]
    x = alt.X("t:T", title=None, axis=alt.Axis(grid=False, labelColor=SLATE, tickColor=LINE, domainColor=LINE, labelFlush=True))
    y = alt.Y("v:Q", title=None, scale=alt.Scale(zero=zero, nice=True, padding=8),
              axis=alt.Axis(format=fmt, gridColor="#EDEFF3", domain=False, ticks=False, labelColor=SLATE, labelPadding=6))
    col = alt.Color("série:N", scale=alt.Scale(domain=noms, range=couleurs),
                    legend=alt.Legend(orient="top-left", title=None, symbolType="stroke", symbolStrokeWidth=3,
                                      labelColor=INK, labelFontSize=12, labelLimit=260, padding=4))
    sel = alt.selection_point(fields=["t"], nearest=True, on="mouseover", clear="mouseout", empty=False)
    lignes = alt.Chart(long).mark_line(strokeWidth=2, strokeJoin="round").encode(x=x, y=y, color=col)
    points = alt.Chart(long).mark_point(filled=True, size=70, stroke=SURF, strokeWidth=2).encode(
        x=x, y=y, color=col).transform_filter(sel)
    infobulle = [alt.Tooltip("t:T", title="Date", format="%d %b %Y")] + [
        alt.Tooltip(f"s{i}:Q", title=n, format=fmt) for i, n in enumerate(noms)]
    regle = alt.Chart(wide).mark_rule(color="#C9CEDA").encode(
        x="t:T", opacity=alt.condition(sel, alt.value(1), alt.value(0)), tooltip=infobulle).add_params(sel)
    return alt.layer(lignes, points, regle).properties(height=hauteur).configure_view(stroke=None).configure(
        background="rgba(0,0,0,0)", font="Hanken Grotesk")


def histogramme(serie, hauteur=200):
    d = serie.rename("n").reset_index()
    d.columns = ["agent", "n"]
    return alt.Chart(d).mark_bar(cornerRadiusEnd=4, color=ACCENT, size=22).encode(
        y=alt.Y("agent:N", title=None, sort="-x", axis=alt.Axis(labelColor=INK, domain=False, ticks=False)),
        x=alt.X("n:Q", title=None, axis=alt.Axis(gridColor="#EDEFF3", domain=False, ticks=False, labelColor=SLATE)),
        tooltip=["agent", "n"]).properties(height=hauteur).configure_view(stroke=None).configure(
        background="rgba(0,0,0,0)", font="Hanken Grotesk")


@st.cache_data
def charger():
    res = json.load(open(os.path.join(OUT, "results.json"), encoding="utf-8"))
    met = json.load(open(os.path.join(OUT, "metrics.json"), encoding="utf-8"))
    pred = pd.read_csv(os.path.join(OUT, "predictions.csv"), parse_dates=["date_decision"])
    tr = pd.read_csv(os.path.join(OUT, "transactions.csv"))
    led = json.load(open(os.path.join(OUT, "ledger.json"), encoding="utf-8"))
    return res, met, pred, tr, led


res, met, pred, tr, led = charger()
d = met["donnees"]

# ------------------------------------------------------------------ en-tête
st.markdown(f"""
<div class="top">
  <div><h1>Agents autonomes · DOT/USD</h1>
  <p>Prévision {met['modele_agents']} → stratégie → risque → exécution, portefeuille virtuel et registre blockchain.
  Simulation : aucun ordre réel n'est passé.</p></div>
  <div class="pill"><span>Projet Deep Learning GSTR S3</span></div>
</div>""", unsafe_allow_html=True)

t_live, t_res, t_prev, t_tx, t_bc, t_mod = st.tabs(["Direct", "Résultats", "Prévisions", "Transactions", "Registre", "Modèle"])

# ------------------------------------------------------------------ DIRECT
with t_live:
    mode = st.radio("Source", ["auto", "rejeu"], horizontal=True, label_visibility="collapsed",
                    format_func=lambda m: "Direct · agents autonomes (prix Kraken)" if m == "auto" else "Rejeu accéléré du test")
    vitesse = st.slider("Secondes par jour de rejeu", 0.1, 3.0, 0.5, 0.1) if mode == "rejeu" else 0.5

    @st.cache_resource
    def demarrer_rejeu(vitesse):
        import live_agent as la
        sim = la.Simulateur(la.Modele(), "rejeu")
        threading.Thread(target=la.boucle_rejeu, args=(sim, threading.Event(), vitesse), daemon=True).start()
        return sim

    def obtenir_sim():
        import live_agent as la
        if mode == "rejeu":
            return demarrer_rejeu(vitesse)
        if not os.path.exists(os.path.join(ICI, "live", "etat_direct.json")):
            return None
        return la.Simulateur(la.Modele(), "direct")  # relit l'état écrit par GitHub Actions

    @st.fragment(run_every=10)
    def direct():
        sim = obtenir_sim()
        if sim is None:
            st.info("Les agents n'ont pas encore tourné. Lance un premier passage : dépôt GitHub → Actions → "
                    "« Agents autonomes DOT/USD » → Run workflow. Ensuite, ils passent seuls à 00:05 UTC puis toutes les 4 h.")
            return
        v = sim.vue()
        cap = v["capital"]
        h = pd.DataFrame(v["histo"])
        derniere = h["t"].iloc[-1] if len(h) else None
        cls = "warn" if v["erreur"] else ("dot" if mode == "auto" else "dot")
        maj = ""
        if derniere:
            try:
                maj = pd.to_datetime(derniere).strftime("%d/%m/%Y %H:%M") + " UTC"
            except Exception:
                maj = str(derniere)
        st.markdown(f'<div style="display:flex;justify-content:flex-end;margin:.2rem 0 .8rem"><div class="pill">'
                    f'<span class="dot {"warn" if v["erreur"] else ""}"></span><b>{"Connexion interrompue" if v["erreur"] else "En service"}</b>'
                    f'<span>Dernière valorisation {maj}</span></div></div>', unsafe_allow_html=True)

        gain, gain_bh = v["valeur"] / cap - 1, v["buy_hold"] / cap - 1
        ecart = v["valeur"] - v["buy_hold"]
        st.markdown('<div class="kpis">' + kpi("Valeur du portefeuille des agents", usd(v["valeur"]),
                                              f"{pct(gain, 2)} depuis le départ", tone(gain), True)
                    + kpi("Buy and Hold", usd(v["buy_hold"]), f"{pct(gain_bh, 2)}", tone(gain_bh))
                    + kpi("Écart agents − Buy and Hold", ("+" if ecart >= 0 else "−") + usd(abs(ecart)), "à ce jour", tone(ecart))
                    + kpi("Prix DOT/USD", "–" if not v["prix"] else f"{fr(v['prix'], 4)} $",
                          pct(v["prix"] / v["prix_depart"] - 1, 2) + " depuis le départ" if v["prix"] and v["prix_depart"] else "")
                    + kpi("Position", f"{v['exposition'] * 100:.0f} %".replace(".", ","),
                          f"{fr(v['unites'], 3)} DOT · {v['n_ordres']} ordres")
                    + '</div>', unsafe_allow_html=True)
        if v["coupe_circuit_actif"]:
            st.error("Coupe-circuit actif : la valeur est passée sous 85 % du capital, les achats sont bloqués.")

        g, p = st.columns([2.1, 1], gap="medium")
        with g:
            with st.container(border=True):
                st.markdown('<div class="card-t">Valeur du portefeuille (en $)</div>', unsafe_allow_html=True)
                if len(h) > 1:
                    h["t"] = pd.to_datetime(h["t"], errors="coerce")
                    st.altair_chart(courbes(h.set_index("t")[["valeur", "bh"]].rename(
                        columns={"valeur": "Agents", "bh": "Buy and Hold"}), 330, ",.2f"), use_container_width=True)
                else:
                    st.caption("En attente des premières valorisations.")
        with p:
            with st.container(border=True):
                st.markdown('<div class="card-t">Dernière décision</div>', unsafe_allow_html=True)
                if v["decisions"]:
                    x = v["decisions"][-1]
                    cfg = v["config"]
                    lo, hi = -0.03, 0.03
                    pos = lambda r: max(0, min(100, (r - lo) / (hi - lo) * 100))
                    sv, sa = cfg["seuil_vente"], cfg["seuil_achat"]
                    st.markdown(
                        f'<span class="badge {x["signal"]}">{x["signal"].capitalize()}</span>'
                        f'<div class="gauge"><div class="z" style="left:0;width:{pos(sv)}%;background:#F6D5D2;border-radius:6px 0 0 6px"></div>'
                        f'<div class="z" style="left:{pos(sa)}%;right:0;background:#CDEBDD;border-radius:0 6px 6px 0"></div>'
                        f'<div class="m" style="left:{pos(x["mu"])}%"></div></div>'
                        f'<div class="gauge-l"><span>Vente sous {pct(sv, 2)}</span><span>Achat dès {pct(sa, 2)}</span></div>'
                        f'<div class="row" style="margin-top:.9rem"><span>Rendement prévu à {v["horizon"]} jours</span><span class="num">{pct(x["mu"], 2)}</span></div>'
                        f'<div class="row"><span>Incertitude (écart des réseaux)</span><span class="num">±{fr(x["sigma"] * 100, 2)} %</span></div>'
                        f'<div class="row"><span>Clôture du</span><span>{x["jour"]}</span></div>'
                        f'<div class="row"><span>Action simulée</span><span>{x["action"].capitalize()}</span></div>'
                        f'<div class="row"><span>Raison</span><span>{x["raison"]}</span></div>',
                        unsafe_allow_html=True)
                else:
                    st.caption("Pas encore de décision.")
                maintenant = datetime.now(timezone.utc)
                prochaine = (maintenant + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                reste = prochaine - maintenant
                st.markdown(f'<div class="row"><span>Prochaine décision</span><span class="num">dans {reste.seconds // 3600} h '
                            f'{(reste.seconds // 60) % 60:02d} min</span></div>', unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<div class="card-t">Prix DOT/USD</div>', unsafe_allow_html=True)
            if len(h) > 1:
                st.altair_chart(courbes(h.set_index("t")[["prix"]].rename(columns={"prix": "DOT/USD"}).rename(
                    columns={"DOT/USD": "Réel"}), 200, ",.4f"), use_container_width=True)

        st.markdown("### Historique des décisions")
        if v["decisions"]:
            dd = pd.DataFrame(v["decisions"][::-1])
            dd = dd.assign(mu=dd.mu * 100, sigma=dd.sigma * 100)[["jour", "signal", "action", "mu", "sigma", "prix", "valeur_apres", "raison"]]
            st.dataframe(dd, use_container_width=True, hide_index=True, column_config={
                "jour": "Clôture du", "signal": "Signal", "action": "Action", "raison": "Raison",
                "mu": st.column_config.NumberColumn("Prévu (%)", format="%+.2f"),
                "sigma": st.column_config.NumberColumn("Incertitude (%)", format="%.2f"),
                "prix": st.column_config.NumberColumn("Prix d'exécution", format="%.4f $"),
                "valeur_apres": st.column_config.NumberColumn("Valeur après ($)", format="%.2f")})
        else:
            st.caption("Aucune décision pour l'instant : les agents décident une fois par jour, à la clôture de 00:00 UTC.")
        ok = v["chaine_valide"]
        (st.success if ok else st.error)(f"Registre blockchain {'valide' if ok else 'ALTÉRÉ'} · {v['n_blocs']} blocs")
        if mode == "auto":
            st.caption("Valeurs mises à jour par GitHub Actions ; la page se rafraîchit toute seule.")

    direct()

# ------------------------------------------------------------------ RÉSULTATS
with t_res:
    st.markdown(f"### Performance sur la période de test · 05/06/2025 → 31/10/2026 ({d['n_test']} jours)")
    s = pd.DataFrame(met["strategies"] + met["ablation"]).drop_duplicates("strategie")
    ordre = ["Agents LSTM", "Règle technique (MM20)", "Buy and Hold", "Agents LSTM sans coupe-circuit"]
    cells = ""
    for i, nom in enumerate(ordre):
        r = s[s.strategie == nom]
        if len(r):
            r = r.iloc[0]
            cells += kpi(nom, usd(r.valeur_finale, 0), f"{pct(r.rendement_total)} · drawdown {pct(r.max_drawdown, 0)}",
                         tone(r.rendement_total), i == 0)
    st.markdown(f'<div class="kpis" style="grid-template-columns:1.3fr repeat(3,1fr)">{cells}</div>', unsafe_allow_html=True)
    cur = {**res["courbes"], **res["courbes_supp"]}
    with st.container(border=True):
        choix = st.multiselect("Courbes affichées", list(cur), default=list(res["courbes"]), label_visibility="collapsed")
        if choix:
            df = pd.DataFrame({n: pd.Series(cur[n]["valeur"], index=pd.to_datetime(cur[n]["dates"])) for n in choix})
            st.altair_chart(courbes(df, 380, ",.0f"), use_container_width=True)
    st.markdown('<div class="note">Les agents limitent la perte surtout grâce au coupe-circuit, qui bloque les achats pendant la chute : '
                'sans lui, ils perdent 36,9 % au lieu de 18,1 %. Ce n\'est pas la preuve d\'une meilleure prévision.</div>',
                unsafe_allow_html=True)
    st.markdown("### Comparaison détaillée")
    aff = s.set_index("strategie")[["valeur_finale", "rendement_total", "max_drawdown", "sharpe", "n_ordres", "frais_totaux", "taux_reussite"]]
    aff = aff.assign(rendement_total=aff.rendement_total * 100, max_drawdown=aff.max_drawdown * 100, taux_reussite=aff.taux_reussite * 100)
    st.dataframe(aff.reset_index(), use_container_width=True, hide_index=True, column_config={
        "strategie": "Stratégie", "valeur_finale": st.column_config.NumberColumn("Valeur finale ($)", format="%.0f"),
        "rendement_total": st.column_config.NumberColumn("Rendement (%)", format="%+.1f"),
        "max_drawdown": st.column_config.NumberColumn("Drawdown max (%)", format="%.1f"),
        "sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
        "n_ordres": "Ordres", "frais_totaux": st.column_config.NumberColumn("Frais ($)", format="%.0f"),
        "taux_reussite": st.column_config.NumberColumn("Réussite (%)", format="%.0f")})
    st.markdown("### Sensibilité aux paramètres de risque et aux frais")
    sn = pd.DataFrame(met["sensibilite"])
    sn = sn.assign(rendement_total=sn.rendement_total * 100, max_drawdown=sn.max_drawdown * 100)
    st.dataframe(sn, use_container_width=True, hide_index=True, column_config={
        "parametre": "Paramètre", "valeur": "Valeur", "valeur_finale": st.column_config.NumberColumn("Valeur finale ($)", format="%.0f"),
        "rendement_total": st.column_config.NumberColumn("Rendement (%)", format="%+.1f"),
        "max_drawdown": st.column_config.NumberColumn("Drawdown max (%)", format="%.1f"),
        "sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"), "n_ordres": "Ordres"})

# ------------------------------------------------------------------ PRÉVISIONS
with t_prev:
    st.markdown("### Erreur de prévision du rendement à 5 jours")
    rows = [dict(ensemble=e, modèle=m, MAE=v["MAE"], RMSE=v["RMSE"], direction=v["direction"] * 100)
            for e, dm in met["prevision"].items() for m, v in dm.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, column_config={
        "ensemble": "Ensemble", "modèle": "Modèle", "MAE": st.column_config.NumberColumn(format="%.4f"),
        "RMSE": st.column_config.NumberColumn(format="%.4f"),
        "direction": st.column_config.NumberColumn("Direction juste (%)", format="%.1f")})
    st.markdown('<div class="note">Sur le test, ni le LSTM ni le GRU ne font mieux que la prévision naïve « moyenne du train » '
                '(RMSE 0,0914 et 0,0919 contre 0,0887) : le modèle n\'a pas de pouvoir prédictif robuste.</div>', unsafe_allow_html=True)
    ens = st.segmented_control("Ensemble", ["test", "validation", "train"], default="test") if hasattr(st, "segmented_control") \
        else st.radio("Ensemble", ["test", "validation", "train"], horizontal=True)
    p = pred[pred.ensemble == (ens or "test")].set_index("date_decision")
    with st.container(border=True):
        st.markdown('<div class="card-t">Rendement à 5 jours : réel et prévu</div>', unsafe_allow_html=True)
        st.altair_chart(courbes(p[["rendement_reel", "prevu_LSTM", "prevu_GRU"]].rename(
            columns={"rendement_reel": "Réel", "prevu_LSTM": "LSTM", "prevu_GRU": "GRU"}), 320, ".0%"), use_container_width=True)
    st.markdown("### Courbes d'apprentissage")
    i = st.slider("Réseau de l'ensemble", 1, len(res["apprentissage"]), 1) - 1
    a = res["apprentissage"][i]
    with st.container(border=True):
        e = pd.DataFrame({"train": a["train"], "validation": a["val"]})
        e.index = pd.to_datetime("2000-01-01") + pd.to_timedelta(range(len(e)), unit="D")
        st.altair_chart(courbes(e, 240, ".2f"), use_container_width=True)
        st.caption("Axe horizontal : époques d'entraînement. L'arrêt anticipé garde le meilleur point de validation.")

# ------------------------------------------------------------------ TRANSACTIONS
with t_tx:
    st.markdown("### Décisions et ordres des agents sur le test")
    actions = sorted(tr.action.unique())
    f = st.multiselect("Action", actions, default=[a for a in actions if a != "RIEN"])
    t2 = tr[tr.action.isin(f)] if f else tr
    st.dataframe(t2, use_container_width=True, hide_index=True, column_config={
        "date_decision": "Décision", "date_execution": "Exécution", "signal": "Signal", "action": "Action",
        "qty": st.column_config.NumberColumn("Quantité", format="%.2f"), "fill": st.column_config.NumberColumn("Prix", format="%.4f"),
        "fee": st.column_config.NumberColumn("Frais ($)", format="%.2f"), "reason": "Raison",
        "mu": st.column_config.NumberColumn("Prévu", format="%+.4f"), "sigma": st.column_config.NumberColumn("Incertitude", format="%.4f"),
        "v_before": st.column_config.NumberColumn("Valeur avant ($)", format="%.0f"),
        "v_after": st.column_config.NumberColumn("Valeur après ($)", format="%.0f")})

# ------------------------------------------------------------------ REGISTRE
with t_bc:
    from blockchain import verify_chain
    chain, diff = led["chain"], led["difficulty"]
    ok, msg = verify_chain(chain, diff)
    (st.success if ok else st.error)(f"{msg.capitalize()} · difficulté {diff} (chaque hash commence par {'0' * diff})")
    a, b = st.columns([1, 2], gap="medium")
    with a:
        with st.container(border=True):
            st.markdown('<div class="card-t">Blocs par agent</div>', unsafe_allow_html=True)
            st.altair_chart(histogramme(pd.Series([x["agent"] for x in chain]).value_counts()), use_container_width=True)
    with b:
        with st.container(border=True):
            st.markdown('<div class="card-t">Comment ça marche</div>', unsafe_allow_html=True)
            st.markdown("Chaque décision d'agent est un bloc contenant le hash du bloc précédent et un nonce (preuve de travail légère). "
                        "Modifier un seul bloc casse tous les hash qui suivent : la vérification le détecte. "
                        "Le fichier `ledger.json` permet au professeur de revérifier la chaîne.")
            st.download_button("Télécharger ledger.json", json.dumps(led, ensure_ascii=False), "ledger.json", "application/json")
    n = st.slider("Derniers blocs affichés", 5, 100, 20)
    st.dataframe(pd.DataFrame([{**{k: x[k] for k in ("index", "sim_time", "agent", "nonce", "hash", "prev_hash")},
                                "données": json.dumps(x["data"], ensure_ascii=False)} for x in chain[-n:]][::-1]),
                 use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ MODÈLE
with t_mod:
    px = pd.Series(res["prix"]["close"], index=pd.to_datetime(res["prix"]["dates"]), name="Réel")
    st.markdown(f"### Prix de clôture DOT/USD · {d['debut']} → {d['fin']}")
    with st.container(border=True):
        st.altair_chart(courbes(px.to_frame(), 300, ",.2f"), use_container_width=True)
    c = st.columns(4)
    for col, (l, vv) in zip(c, [("Jours", d["n_jours"]), ("Entraînement", d["n_train"]), ("Validation", d["n_val"]), ("Test", d["n_test"])]):
        col.metric(l, vv)
    st.markdown("### Recherche d'hyperparamètres")
    hp = pd.DataFrame([dict(famille=f, **r) for f, L in met["recherche_hyperparametres"].items() for r in L])
    st.dataframe(hp, use_container_width=True, hide_index=True, column_config={
        "famille": "Famille", "n_hid": "Unités", "l2": "L2", "rmse_val": st.column_config.NumberColumn("RMSE validation", format="%.4f")})
    st.caption(f"Retenu : {met['modele_agents']}, {met['retenu']['n_hid']} unités, L2 = {met['retenu']['l2']}. "
               f"Gradients vérifiés par différences finies : erreur relative {met['verification_gradient']['LSTM']:.1e} (LSTM), "
               f"{met['verification_gradient']['GRU']:.1e} (GRU).")
    st.markdown("### Configuration des agents")
    st.json(met["configuration"], expanded=False)
    eda = os.path.join(OUT, "eda")
    if os.path.isdir(eda):
        st.markdown("### Analyse exploratoire")
        imgs = sorted(x for x in os.listdir(eda) if x.endswith(".png"))
        cols = st.columns(2)
        for i, fn in enumerate(imgs):
            cols[i % 2].image(os.path.join(eda, fn), use_container_width=True)
