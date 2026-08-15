"""
dashboard/app.py
==================
"Treasury Control Tower" — a live, self-contained Streamlit application.

Unlike a typical portfolio dashboard that replays a pre-computed CSV, this
app:

  1. Bootstraps its own data + trains its own model on first load if no
     artifacts exist yet (via `src/bootstrap.py`) — so it deploys cleanly
     to a fresh container with nothing pre-committed to the repo.
  2. Runs a genuine incremental streaming pipeline: `LiveTransactionSimulator`
     emits one transaction at a time, `StreamingFeatureStore` computes its
     feature vector from bounded rolling state (not a batch dataframe),
     and the persisted XGBoost model scores it — the same three steps a
     real inbound-payment risk service would perform.
  3. Evolves the currency float pools tick-by-tick under a live (s, S)
     inventory policy (`LivePoolBook`): pools actually deplete as
     transactions draw them down and actually replenish after a simulated
     lead time, so the reorder-point crossings and rebalancing alerts you
     see are computed live, not hardcoded.

Run locally:
    streamlit run dashboard/app.py

Deploy: push this repo to GitHub and point Streamlit Community Cloud at
`dashboard/app.py` — no manual pipeline run required, see README.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(ROOT, "data"))
sys.path.insert(0, SRC)

from feature_engineering import FEATURE_COLUMNS  # noqa: E402
from liquidity_optimizer import LivePoolBook, PoolParameters  # noqa: E402
from streaming_engine import LiveTransactionSimulator, StreamingFeatureStore  # noqa: E402

st.set_page_config(
    page_title="Treasury Control Tower",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================================
# Design tokens — "Treasury Desk" theme, light/corporate variant.
# A clean, near-white workspace (not stark white — a soft cool-gray page
# behind pure-white cards, the same layering convention Stripe/Linear-style
# fintech UIs use so cards read as "raised" via a hairline border + a
# barely-there shadow rather than a heavy dark chrome). The same two-color
# signature carries over from the dark variant — teal for healthy/liquid,
# amber for scarcity, rose reserved strictly for risk — just recalibrated
# to hold contrast on white instead of black. Numerals stay monospace
# throughout: tabular figures that align digit-for-digit is how trading/
# treasury terminals present numbers, independent of light or dark mode.
# ======================================================================
BG = "#F5F7FA"
PANEL = "#FFFFFF"
PANEL_ALT = "#F1F4F9"
BORDER = "#E2E7F0"
TEXT = "#1B2432"
MUTED = "#6B7A90"
TEAL = "#0F9D8C"      # healthy / approved / liquidity
GOLD = "#C2760C"      # scarcity / flagged / value
ROSE = "#D6425C"      # risk / blocked / stockout
GRADIENT = f"linear-gradient(90deg, {TEAL}, {GOLD})"
CARD_SHADOW = "0 1px 2px rgba(16,24,40,0.04), 0 1px 3px rgba(16,24,40,0.06)"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; }}
    .stApp {{ background-color: {BG}; }}
    section[data-testid="stSidebar"] {{
        background-color: {PANEL}; border-right: 1px solid {BORDER};
    }}
    h1, h2, h3 {{
        font-family: 'Space Grotesk', sans-serif !important;
        color: {TEXT} !important; letter-spacing: 0.2px;
    }}
    .tt-header {{
        display: flex; align-items: baseline; gap: 12px; margin-bottom: 2px;
    }}
    .tt-header h1 {{ margin: 0; font-size: 1.9rem; }}
    .tt-rule {{
        height: 3px; width: 64px; background: {GRADIENT}; border-radius: 3px;
        margin: 6px 0 18px 0;
    }}
    .tt-caption {{ color: {MUTED}; font-size: 0.92rem; margin-bottom: 4px; }}

    div[data-testid="stMetric"] {{
        background-color: {PANEL}; border: 1px solid {BORDER};
        border-radius: 10px; padding: 12px 16px 12px 16px;
        box-shadow: {CARD_SHADOW};
    }}
    div[data-testid="stMetricLabel"] {{ color: {MUTED}; font-size: 0.8rem; }}
    div[data-testid="stMetricValue"] {{
        font-family: 'IBM Plex Mono', monospace !important; color: {TEXT};
        font-size: 1.5rem !important;
    }}

    .pulse-dot {{
        height: 9px; width: 9px; border-radius: 50%; display: inline-block;
        margin-right: 7px; position: relative; top: -1px;
    }}
    .pulse-live {{ background: {TEAL}; box-shadow: 0 0 0 0 rgba(15,157,140,0.5); animation: pulse 1.6s infinite; }}
    .pulse-idle {{ background: {MUTED}; }}
    @keyframes pulse {{
        0%   {{ box-shadow: 0 0 0 0 rgba(15,157,140,0.45); }}
        70%  {{ box-shadow: 0 0 0 9px rgba(15,157,140,0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(15,157,140,0); }}
    }}
    div[data-testid="stDataFrame"] {{
        border: 1px solid {BORDER}; border-radius: 10px; box-shadow: {CARD_SHADOW};
    }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {BORDER}; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent; border-radius: 8px 8px 0 0; color: {MUTED};
        font-family: 'IBM Plex Sans', sans-serif; font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{ color: {TEAL} !important; }}

    /* Streamlit's light-theme defaults leave a few elements on a white-
       on-white or default-blue styling that clashes with this palette —
       pinned explicitly rather than left to fight the base theme. */
    section[data-testid="stSidebar"] * {{ color: {TEXT}; }}
    section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] small {{ color: {MUTED}; }}
</style>
""", unsafe_allow_html=True)


# ======================================================================
# Bootstrap: ensure artifacts exist (fresh container-safe)
# ======================================================================
@st.cache_resource(show_spinner=False)
def ensure_artifacts() -> dict:
    required = [
        "transactions.csv", "model_metrics.json", "fraud_engine.joblib",
        "corridor_frequency.json", "pool_parameters.json", "liquidity_state.json",
    ]
    if all(os.path.exists(os.path.join(DATA_DIR, r)) for r in required):
        with open(os.path.join(DATA_DIR, "model_metrics.json")) as f:
            return {"metrics": json.load(f), "bootstrapped_now": False}
    from bootstrap import run_bootstrap
    result = run_bootstrap(DATA_DIR)
    return {"metrics": result["metrics"], "bootstrapped_now": True}


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(os.path.join(DATA_DIR, "fraud_engine.joblib"))


@st.cache_resource(show_spinner=False)
def load_static_context():
    corridor_freq = json.load(open(os.path.join(DATA_DIR, "corridor_frequency.json")))
    fi_path = os.path.join(DATA_DIR, "feature_importance.csv")
    fi = pd.read_csv(fi_path) if os.path.exists(fi_path) else pd.DataFrame(columns=["feature", "importance"])
    pool_params_raw = json.load(open(os.path.join(DATA_DIR, "pool_parameters.json")))
    return corridor_freq, fi, pool_params_raw


with st.spinner("Bootstrapping synthetic data, training fraud model, solving liquidity pools…"):
    bootstrap_info = ensure_artifacts()

engine = load_model()
corridor_freq, feature_importance_df, pool_params_raw = load_static_context()
train_metrics = bootstrap_info["metrics"]


# ======================================================================
# Session state: this IS the live system's state
# ======================================================================
def _init_session():
    if "feature_store" not in st.session_state:
        st.session_state.feature_store = StreamingFeatureStore(corridor_frequency=corridor_freq)
    if "simulator" not in st.session_state:
        st.session_state.simulator = LiveTransactionSimulator(n_users=350, seed=None, fraud_probability=0.05)
    if "pool_book" not in st.session_state:
        pool_params = [PoolParameters(**d) for d in pool_params_raw]
        st.session_state.pool_book = LivePoolBook(pool_params, replenishment_lead_ticks=6)
    if "log" not in st.session_state:
        st.session_state.log = pd.DataFrame(columns=[
            "Transaction_ID", "Timestamp", "User_ID", "Source_Currency", "Target_Currency",
            "Amount_USD", "Velocity_1H", "amount_zscore", "geo_mismatch",
            "is_fraud", "fraud_pattern", "fraud_probability", "action", "latency_ms",
        ])
    if "liquidity_snapshot" not in st.session_state:
        st.session_state.liquidity_snapshot = None
    if "tick_count" not in st.session_state:
        st.session_state.tick_count = 0
    if "live_mode" not in st.session_state:
        st.session_state.live_mode = False


_init_session()
MAX_LOG_ROWS = 1500


def advance(n_txns: int):
    """Performs one simulation tick: generate n_txns live transactions,
    run each through the streaming feature store, score the batch with
    the persisted model, evolve the liquidity pools by the resulting
    outflows, and append to the session's rolling log."""
    sim: LiveTransactionSimulator = st.session_state.simulator
    store: StreamingFeatureStore = st.session_state.feature_store
    pool_book: LivePoolBook = st.session_state.pool_book

    now = datetime.now()
    raw_txns = sim.next_batch(n_txns, start_time=now, seconds_between=0.4)
    featured_rows = [store.process(t) for t in raw_txns]
    batch_df = pd.DataFrame(featured_rows)

    t0 = time.perf_counter()
    X = batch_df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0)
    proba = engine.model.predict_proba(X)[:, 1]
    latency_ms = (time.perf_counter() - t0) * 1000 / max(len(batch_df), 1)
    actions = engine.policy.classify(proba)

    batch_df["fraud_probability"] = proba
    batch_df["action"] = actions
    batch_df["latency_ms"] = round(latency_ms, 4)

    keep_cols = list(st.session_state.log.columns)
    if len(st.session_state.log) == 0:
        st.session_state.log = batch_df[keep_cols].copy()
    else:
        st.session_state.log = pd.concat(
            [batch_df[keep_cols], st.session_state.log], ignore_index=True
        )
    st.session_state.log = st.session_state.log.head(MAX_LOG_ROWS)

    outflows = batch_df.groupby("Source_Currency")["Amount_USD"].sum().to_dict()
    st.session_state.liquidity_snapshot = pool_book.tick(outflows)
    st.session_state.tick_count += 1


# ======================================================================
# Sidebar
# ======================================================================
st.sidebar.markdown(
    f"<div style='font-family:Space Grotesk; font-size:1.15rem; color:{TEXT}; "
    f"font-weight:600;'>◈ Treasury Control Tower</div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    f"<div style='color:{MUTED}; font-size:0.85rem; margin-bottom:16px;'>"
    "Live fraud risk × multi-currency liquidity simulation</div>",
    unsafe_allow_html=True,
)

live_toggle = st.sidebar.toggle("Live simulation", value=st.session_state.live_mode)
st.session_state.live_mode = live_toggle
batch_size = st.sidebar.slider("Transactions per tick", 5, 60, 15, step=5)
tick_interval = st.sidebar.slider("Tick interval (seconds)", 0.5, 4.0, 1.5, step=0.5)

col_a, col_b = st.sidebar.columns(2)
if col_a.button("⟳ Step once"):
    advance(batch_size)
if col_b.button("↺ Reset session"):
    for k in ["feature_store", "simulator", "pool_book", "log", "liquidity_snapshot", "tick_count"]:
        del st.session_state[k]
    _init_session()
    st.rerun()

status = "pulse-live" if st.session_state.live_mode else "pulse-idle"
label = "LIVE" if st.session_state.live_mode else "PAUSED"
st.sidebar.markdown(
    f"<span class='pulse-dot {status}'></span>"
    f"<span style='font-family:IBM Plex Mono; color:{TEXT}; font-size:0.85rem;'>{label}</span>"
    f"&nbsp;&nbsp;<span style='color:{MUTED}; font-family:IBM Plex Mono; font-size:0.8rem;'>"
    f"tick {st.session_state.tick_count} · {len(st.session_state.log)} txns in session</span>",
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")
with st.sidebar.expander("How this is 'live'"):
    st.markdown(
        "- Each tick, `LiveTransactionSimulator` emits new transactions "
        "(never pre-generated).\n"
        "- `StreamingFeatureStore` computes every feature — velocity, "
        "personalized z-score, device fan-out — from bounded rolling state "
        "**as each transaction arrives**, not from a batch dataframe.\n"
        "- The persisted XGBoost model scores the live batch directly.\n"
        "- Currency pools actually deplete from live outflows and "
        "replenish under an (s, S) policy after a simulated lead time."
    )
with st.sidebar.expander("Model quality caveat"):
    _n_bootstrap_rows = bootstrap_info["metrics"]["n_train"] + bootstrap_info["metrics"]["n_test"]
    st.markdown(
        f"Trained on a demo-scale synthetic set (~{_n_bootstrap_rows:,} rows) "
        "for fast cold starts. Metrics below reflect that scale; a larger "
        "offline run (`python src/data_generator.py` at full scale) is used "
        "for the headline numbers in the README."
    )

if st.session_state.live_mode:
    advance(batch_size)


# ======================================================================
# Header
# ======================================================================
st.markdown(
    "<div class='tt-header'><h1>Treasury &amp; Risk Control Tower</h1></div>"
    "<div class='tt-rule'></div>"
    "<div class='tt-caption'>Real-time multi-currency liquidity optimization "
    "+ fraud/AML risk scoring — live simulation, not a replayed dataset.</div>",
    unsafe_allow_html=True,
)

log = st.session_state.log
liq_snapshot = st.session_state.liquidity_snapshot
if liq_snapshot is None:
    advance(batch_size)
    liq_snapshot = st.session_state.liquidity_snapshot

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Live transactions", f"{len(log):,}")
session_recall = (
    log.loc[log["is_fraud"] == 1, "action"].ne("Approved").mean()
    if (log["is_fraud"] == 1).any() else float("nan")
)
k2.metric("Session fraud catch rate", f"{session_recall:.0%}" if session_recall == session_recall else "—")
k3.metric("Trained-model precision", f"{train_metrics['precision_at_policy']:.1%}")
avg_latency = log["latency_ms"].mean() if len(log) else 0.0
k4.metric("Avg inference latency", f"{avg_latency:.3f} ms")
pools_at_risk = int(liq_snapshot["needs_rebalancing"].sum())
k5.metric("Pools below reorder point", pools_at_risk,
          delta=None if pools_at_risk == 0 else "rebalancing", delta_color="inverse")

st.markdown("")

tab_overview, tab_feed, tab_model, tab_liquidity = st.tabs(
    ["Overview", "Live Feed", "Model Confidence", "Liquidity & Pricing"]
)

# ======================================================================
# TAB 1 — Overview
# ======================================================================
with tab_overview:
    c1, c2 = st.columns([1.35, 1])

    with c1:
        st.subheader("Float pool levels vs. reorder point")
        fig = go.Figure()
        for _, row in liq_snapshot.iterrows():
            color = ROSE if row["needs_rebalancing"] else TEAL
            fig.add_trace(go.Bar(
                x=[row["currency"]], y=[row["current_level"]],
                marker_color=color, showlegend=False,
                hovertemplate=f"{row['currency']}<br>Level: %{{y:,.0f}}<extra></extra>",
            ))
        fig.add_trace(go.Scatter(
            x=liq_snapshot["currency"], y=liq_snapshot["reorder_point"], mode="markers",
            marker=dict(symbol="line-ew", size=26, color=GOLD, line=dict(width=3, color=GOLD)),
            name="Reorder point",
        ))
        fig.update_layout(
            template="plotly_white", paper_bgcolor=PANEL, plot_bgcolor=PANEL,
            font=dict(color=TEXT, family="IBM Plex Sans"),
            height=360, margin=dict(t=10, l=10, r=10, b=10),
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig, width='stretch')
        st.caption("Bars = live pool balance, updated every tick from real outflows. "
                   "Gold tick = reorder point from the (s, S) newsvendor solve.")

    with c2:
        st.subheader("Session action mix")
        if len(log):
            dist = log["action"].value_counts(normalize=True)
        else:
            dist = pd.Series(dtype=float)
        colors = {"Approved": TEAL, "Flagged for Compliance Review": GOLD, "Blocked": ROSE}
        fig2 = go.Figure(go.Pie(
            labels=dist.index, values=dist.values, hole=0.58,
            marker=dict(colors=[colors.get(label, MUTED) for label in dist.index]),
        ))
        fig2.update_layout(
            template="plotly_white", paper_bgcolor=PANEL, height=360,
            font=dict(color=TEXT, family="IBM Plex Sans"),
            margin=dict(t=10, l=10, r=10, b=10),
            legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(fig2, width='stretch')

    st.subheader("Fee multiplier by currency (live)")
    fig_fee = px.bar(
        liq_snapshot.sort_values("fee_multiplier", ascending=False),
        x="currency", y="fee_multiplier", color="needs_rebalancing",
        color_discrete_map={True: ROSE, False: TEAL},
    )
    fig_fee.update_layout(
        template="plotly_white", paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="IBM Plex Sans"),
        height=280, margin=dict(t=10, l=10, r=10, b=10), showlegend=False,
    )
    st.plotly_chart(fig_fee, width='stretch')

# ======================================================================
# TAB 2 — Live feed
# ======================================================================
with tab_feed:
    st.subheader("Live transaction feed")
    st.caption("Newest first. Each row was scored the instant it was generated — "
               "features computed from rolling state, not looked up from a table.")

    if len(log) == 0:
        st.info("No transactions yet — toggle **Live simulation** or click **Step once**.")
    else:
        display_cols = [
            "Transaction_ID", "Timestamp", "User_ID", "Source_Currency", "Target_Currency",
            "Amount_USD", "Velocity_1H", "amount_zscore", "geo_mismatch",
            "fraud_probability", "action",
        ]

        def _style_action(val):
            color = {"Approved": TEAL, "Flagged for Compliance Review": GOLD, "Blocked": ROSE}.get(val, TEXT)
            return f"color: {color}; font-weight: 600"

        styled = log[display_cols].head(200).style.map(_style_action, subset=["action"]).format({
            "fraud_probability": "{:.4f}", "Amount_USD": "${:,.2f}", "amount_zscore": "{:.2f}",
        })
        st.dataframe(styled, width='stretch', height=480)

    fcol1, fcol2, fcol3 = st.columns(3)
    n_flagged = int((log["action"] != "Approved").sum()) if len(log) else 0
    n_true_fraud = int(log["is_fraud"].sum()) if len(log) else 0
    n_patterns = log.loc[log["is_fraud"] == 1, "fraud_pattern"].nunique() if len(log) else 0
    fcol1.metric("Flagged/Blocked this session", n_flagged)
    fcol2.metric("True fraud generated", n_true_fraud)
    fcol3.metric("Distinct fraud patterns seen", n_patterns)

# ======================================================================
# TAB 3 — Model confidence
# ======================================================================
with tab_model:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Live fraud-probability distribution")
        if len(log) >= 5:
            fig3 = px.histogram(
                log, x="fraud_probability", color="is_fraud", nbins=40,
                barmode="overlay", opacity=0.75,
                color_discrete_map={0: TEAL, 1: ROSE},
                labels={"is_fraud": "True label"},
            )
            fig3.add_vline(x=engine.policy.t_flag, line_dash="dash", line_color=GOLD, annotation_text="t_flag")
            fig3.add_vline(x=engine.policy.t_block, line_dash="dash", line_color=ROSE, annotation_text="t_block")
            fig3.update_layout(
                template="plotly_white", paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                font=dict(color=TEXT, family="IBM Plex Sans"),
                height=400, margin=dict(t=10, l=10, r=10, b=10),
            )
            st.plotly_chart(fig3, width='stretch')
        else:
            st.info("Accumulating live transactions — advance the simulation to populate this chart.")

    with c2:
        st.subheader("Top model features (trained offline)")
        fig4 = px.bar(
            feature_importance_df.head(12).sort_values("importance"),
            x="importance", y="feature", orientation="h",
            color="importance", color_continuous_scale=[PANEL_ALT, TEAL],
        )
        fig4.update_layout(
            template="plotly_white", paper_bgcolor=PANEL, plot_bgcolor=PANEL,
            font=dict(color=TEXT, family="IBM Plex Sans"),
            height=400, margin=dict(t=10, l=10, r=10, b=10), coloraxis_showscale=False,
        )
        st.plotly_chart(fig4, width='stretch')
        st.caption("Importance is diagnostic, not causal — a sanity check that the model "
                   "leans on genuine behavioral signal rather than a generator artifact.")

    st.subheader("Trained model performance (offline holdout)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC", f"{train_metrics['roc_auc']:.3f}")
    m2.metric("PR-AUC", f"{train_metrics['pr_auc']:.3f}")
    m3.metric("Recall @ policy", f"{train_metrics['recall_at_policy']:.1%}")
    m4.metric("Precision @ policy", f"{train_metrics['precision_at_policy']:.1%}")

# ======================================================================
# TAB 4 — Liquidity & pricing
# ======================================================================
with tab_liquidity:
    st.subheader("(s, S) newsvendor state — live")
    st.dataframe(
        liq_snapshot.set_index("currency")[[
            "current_level", "reorder_point", "safety_stock",
            "service_level", "scarcity_ratio", "fee_multiplier",
            "needs_rebalancing", "replenishment_pending",
        ]].style.format({
            "current_level": "{:,.0f}", "reorder_point": "{:,.0f}",
            "safety_stock": "{:,.0f}", "service_level": "{:.2%}",
            "scarcity_ratio": "{:.2f}", "fee_multiplier": "{:.2f}x",
        }).map(lambda v: f"color:{ROSE}; font-weight:700" if v is True else "",
               subset=["needs_rebalancing"]),
        width='stretch',
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Dynamic fee curve vs. live pool scarcity")
        x = np.linspace(-4, 4, 200)
        y = 1.0 + (2.5 - 1.0) * (1 / (1 + np.exp(-(-x * 2.0))))
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=TEAL, width=3), name="Fee curve"))
        fig6.add_trace(go.Scatter(
            x=liq_snapshot["scarcity_ratio"], y=liq_snapshot["fee_multiplier"], mode="markers+text",
            text=liq_snapshot["currency"], textposition="top center",
            marker=dict(size=12, color=[ROSE if r else TEAL for r in liq_snapshot["needs_rebalancing"]]),
            name="Pools (live)",
        ))
        fig6.update_layout(
            template="plotly_white", paper_bgcolor=PANEL, plot_bgcolor=PANEL,
            font=dict(color=TEXT, family="IBM Plex Sans"),
            height=400, margin=dict(t=10, l=10, r=10, b=10),
            xaxis_title="Scarcity ratio  (level − reorder point) / safety stock",
            yaxis_title="Fee multiplier",
        )
        st.plotly_chart(fig6, width='stretch')

    with c2:
        st.subheader("Rebalancing alerts")
        alerts = liq_snapshot[liq_snapshot["needs_rebalancing"]]
        if alerts.empty:
            st.success("All float pools are above their reorder point.")
        else:
            for _, row in alerts.iterrows():
                shortfall = row["reorder_point"] - row["current_level"]
                pending_note = " Replenishment already in transit." if row["replenishment_pending"] else ""
                st.error(
                    f"**{row['currency']} pool below reorder point** — current "
                    f"{row['current_level']:,.0f} vs. reorder point {row['reorder_point']:,.0f} "
                    f"(short {shortfall:,.0f}). Fee multiplier: **{row['fee_multiplier']:.2f}x**."
                    f"{pending_note}"
                )

st.markdown("")
st.markdown(
    f"<div style='color:{MUTED}; font-size:0.78rem; text-align:center; padding-top:8px; "
    f"border-top:1px solid {BORDER};'>Synthetic simulation — no real financial data. "
    "Fraud actions come from a persisted XGBoost model + two-threshold policy layer; "
    "liquidity state evolves under a live (s, S) inventory policy.</div>",
    unsafe_allow_html=True,
)

if st.session_state.live_mode:
    time.sleep(tick_interval)
    st.rerun()