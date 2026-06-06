"""
Rossmann Sales Forecasting — Streamlit application (cloud-friendly build)
=========================================================================
Model results (SimpleRNN, LSTM, SARIMA) are pre-computed by `precompute.py`
and stored in results.json, so this app needs only lightweight libraries and
deploys cleanly on Streamlit Community Cloud. The dashboard stays fully
interactive: pick a model, change the forecast horizon, and run the promotion
what-if.

Run with:   streamlit run app.py
To regenerate the model results (e.g. after adding the real Kaggle data):
            python precompute.py
"""
import os, json, subprocess
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

SEED = 7
ACCENT = "#2563eb"

st.set_page_config(page_title="Rossmann Sales Forecasting", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background: #f7f9fc; }
    h1, h2, h3 { color: #0f172a; font-family: 'Inter', system-ui, sans-serif; }
    .metric-card { background:#fff; border-radius:16px; padding:18px 20px;
        box-shadow:0 1px 3px rgba(15,23,42,.08); border:1px solid #eef2f7; }
    .metric-card .label { color:#64748b; font-size:.80rem; text-transform:uppercase;
        letter-spacing:.05em; margin-bottom:4px; }
    .metric-card .value { color:#0f172a; font-size:1.7rem; font-weight:700; }
    .metric-card .sub { color:#16a34a; font-size:.8rem; }
    .winner { background:#ecfdf5 !important; border-color:#a7f3d0 !important; }
    [data-testid="stSidebar"] { background:#0f172a; }
    [data-testid="stSidebar"] * { color:#e2e8f0 !important; }
    [data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2 { color:#fff !important; }
</style>
""", unsafe_allow_html=True)


def card(label, value, sub="", winner=False):
    cls = "metric-card winner" if winner else "metric-card"
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(f'<div class="{cls}"><div class="label">{label}</div>'
                f'<div class="value">{value}</div>{sub_html}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Data + precomputed results
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading data…")
def load_data():
    if not (os.path.exists("train.csv") and os.path.exists("store.csv")):
        subprocess.run(["python", "generate_data.py"], check=True)
    store = pd.read_csv("store.csv")
    train = pd.read_csv("train.csv", parse_dates=["Date"])
    return store, train


@st.cache_data(show_spinner=False)
def load_results():
    with open("results.json") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def daily_series(train):
    g = train.groupby("Date")
    d = pd.DataFrame({"Sales": g.apply(lambda x: x.loc[x.Open == 1, "Sales"].mean())}) \
        .reset_index().sort_values("Date")
    d["Sales"] = d["Sales"].interpolate().bfill().ffill()
    return d


store, train = load_data()
R = load_results()
meta = R["meta"]

with st.sidebar:
    st.title("📈 Sales Forecasting")
    st.caption("Rossmann Store Sales — RNN · LSTM · SARIMA")
    st.markdown("---")
    st.metric("Stores in dataset", f"{store.Store.nunique()}")
    st.metric("Days of history", f"{len(meta['hist_tail_dates']) and meta['split'] + len(meta['actual'])}")
    using_real = os.path.exists("train.csv") and train.shape[0] > 200000
    st.caption("✅ Real Kaggle data" if using_real else "🧪 Demo data (same schema as Kaggle)")
    st.markdown("---")
    st.caption("Model results are pre-computed for fast, lightweight deployment. "
               "Re-run `python precompute.py` to refresh them.")

st.title("Rossmann Store Sales — Forecasting Dashboard")
tab1, tab2, tab3 = st.tabs(["🔍 Overview & EDA", "🏁 Model Comparison", "🔮 Forecast & Simulate"])

# ----------------------------------------------------------------------------
# TAB 1 — EDA
# ----------------------------------------------------------------------------
with tab1:
    open_df = train[train.Open == 1]
    c1, c2, c3, c4 = st.columns(4)
    with c1: card("Avg sales / open store", f"€{open_df.Sales.mean():,.0f}")
    with c2: card("Avg customers / day", f"{open_df.Customers.mean():,.0f}")
    lift = open_df.groupby("Promo").Sales.mean()
    with c3: card("Promotional uplift", f"+{(lift[1]/lift[0]-1)*100:,.1f}%", "promo vs no-promo")
    with c4: card("Open-day rate", f"{train.Open.mean()*100:,.0f}%")

    st.markdown("### Average sales per open store over time")
    d = daily_series(train)
    fig = px.line(d, x="Date", y="Sales", color_discrete_sequence=[ACCENT])
    fig.update_layout(height=330, margin=dict(t=10, b=10), yaxis_title="Avg sales")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("A gentle upward trend, a yearly cycle peaking each December, and a strong "
               "weekly oscillation — the three signals the models must learn.")

    cA, cB = st.columns(2)
    with cA:
        st.markdown("### Weekly seasonality")
        names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow = open_df.assign(d=open_df.Date.dt.dayofweek).groupby("d").Sales.mean()
        fig = px.bar(x=names, y=dow.values, color=dow.values, color_continuous_scale="Blues")
        fig.update_layout(height=300, margin=dict(t=10, b=10), coloraxis_showscale=False,
                          xaxis_title="", yaxis_title="Avg sales")
        st.plotly_chart(fig, use_container_width=True)
    with cB:
        st.markdown("### Promotion effect")
        fig = px.box(open_df.sample(min(8000, len(open_df)), random_state=SEED),
                     x="Promo", y="Sales", color="Promo",
                     color_discrete_sequence=["#94a3b8", "#16a34a"])
        fig.update_layout(height=300, margin=dict(t=10, b=10), showlegend=False,
                          xaxis_title="Promotion running", yaxis_title="Sales")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Average sales by store type")
    m = open_df.merge(store[["Store", "StoreType"]], on="Store")
    byt = m.groupby("StoreType").Sales.mean().reset_index()
    fig = px.bar(byt, x="StoreType", y="Sales", color="Sales", color_continuous_scale="Viridis")
    fig.update_layout(height=300, margin=dict(t=10, b=10), coloraxis_showscale=False, yaxis_title="Avg sales")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 2 — Model comparison
# ----------------------------------------------------------------------------
with tab2:
    st.markdown("### Model comparison")
    st.caption("All three models are scored on the same held-out 20% test period "
               "(the most recent days). Lower error is better.")

    rows = [{"Model": ("SARIMA(1,1,1)(1,1,1)₇" if k == "SARIMA" else k), "_key": k,
             **R["models"][k]["metrics"]} for k in ["SARIMA", "SimpleRNN", "LSTM"]]
    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    best = results.iloc[0]["Model"]

    cols = st.columns(3)
    for col, (_, r) in zip(cols, results.iterrows()):
        with col:
            card(r["Model"], f"{r['MAPE']:.2f}%",
                 f"RMSE {r['RMSE']:,.0f} · MAE {r['MAE']:,.0f}", winner=(r["Model"] == best))
    st.success(f"🏆 Best model: **{best}** — lowest error on the held-out period. "
               f"Seasonal-naïve baseline RMSE ≈ {R['baseline_rmse']:,.0f} for reference.")

    st.markdown("### Predicted vs actual — test period")
    td = pd.to_datetime(meta["test_dates"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=td, y=meta["actual"], name="Actual",
                             line=dict(color="#0f172a", width=3)))
    fig.add_trace(go.Scatter(x=td, y=R["models"]["SimpleRNN"]["pred"], name="SimpleRNN",
                             line=dict(color="#2563eb", width=1.6)))
    fig.add_trace(go.Scatter(x=td, y=R["models"]["LSTM"]["pred"], name="LSTM",
                             line=dict(color="#16a34a", width=1.6)))
    fig.add_trace(go.Scatter(x=td, y=R["models"]["SARIMA"]["pred"], name="SARIMA",
                             line=dict(color="#f59e0b", width=1.6, dash="dash")))
    fig.update_layout(height=420, margin=dict(t=10), yaxis_title="Avg sales",
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📉 Training curves (neural networks)"):
        cc1, cc2 = st.columns(2)
        for cc, name in [(cc1, "SimpleRNN"), (cc2, "LSTM")]:
            with cc:
                cv = R["curves"][name]
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=cv["loss"], name="train"))
                fig.add_trace(go.Scatter(y=cv["val_loss"], name="validation"))
                fig.update_layout(title=name, height=280, margin=dict(t=30),
                                  xaxis_title="epoch", yaxis_title="MSE (scaled)")
                st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 3 — Forecast & simulate
# ----------------------------------------------------------------------------
with tab3:
    st.markdown("### Forecast future sales — and simulate a promotion")
    c1, c2, c3 = st.columns(3)
    with c1:
        kind = st.selectbox("Model", ["SimpleRNN", "LSTM"])
    with c2:
        horizon = st.slider("Forecast horizon (days)", 7, meta["horizon"], 30, 7)
    with c3:
        promo = st.slider("Promotion scenario — share of stores on promo",
                          0.0, 1.0, round(meta["p0"], 2), 0.01,
                          help="What-if lever. The neural-net forecast is computed at the "
                               "historical promo level; this scenario then applies the "
                               "empirically measured promotional uplift.")

    p0, u = meta["p0"], meta["measured_uplift"]
    base = np.array(R["forecasts"][kind][:horizon])
    factor = (1 + u * promo) / (1 + u * p0)
    fc = base * factor
    fdates = pd.to_datetime(meta["forecast_dates"][:horizon])

    k1, k2, k3 = st.columns(3)
    with k1: card("Forecast total", f"€{fc.sum():,.0f}", f"next {horizon} days")
    with k2: card("Daily average", f"€{fc.mean():,.0f}")
    with k3: card("Scenario vs typical", f"{(factor-1)*100:+.1f}%",
                  f"promo share {promo:.0%} vs {p0:.0%}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pd.to_datetime(meta["hist_tail_dates"]), y=meta["hist_tail_sales"],
                             name="History", line=dict(color="#94a3b8")))
    fig.add_trace(go.Scatter(x=fdates, y=base, name="Expected (typical promo)",
                             line=dict(color="#94a3b8", width=1.6, dash="dot")))
    fig.add_trace(go.Scatter(x=fdates, y=fc, name=f"{kind} — scenario",
                             line=dict(color=ACCENT, width=2.5)))
    fig.update_layout(height=420, margin=dict(t=10), yaxis_title="Avg sales",
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"The neural network forecasts sales one day at a time (recursive forecasting) "
               f"at the historical promotion level. The **scenario** line then applies the "
               f"measured **+{u*100:.1f}%** per-store promotional uplift. Slide the promo share "
               f"to compare.")

    out = pd.DataFrame({"Date": fdates.strftime("%Y-%m-%d"), "Forecast": fc.round(1)})
    st.download_button("⬇️ Download forecast (CSV)", out.to_csv(index=False).encode(),
                       file_name="sales_forecast.csv", mime="text/csv")
