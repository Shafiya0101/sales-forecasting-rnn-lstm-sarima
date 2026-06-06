"""
Rossmann Sales Forecasting — Streamlit application
==================================================
Interactive companion to the notebook. Three tabs:
  1. Overview & EDA      – KPIs and the key sales patterns
  2. Model Comparison    – train SimpleRNN / LSTM / SARIMA and score them
  3. Forecast & Simulate – forecast future days and simulate a promotion

Run with:   streamlit run app.py
The app reads store.csv / train.csv from this folder. If they are missing it
generates a realistic stand-in with the identical Rossmann schema, so it runs
out of the box. Download the real data from:
    https://www.kaggle.com/competitions/rossmann-store-sales/data
"""
import os, warnings, subprocess
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

SEED = 7
FEATURES = ["Sales", "dow_sin", "dow_cos", "m_sin", "m_cos", "PromoShare", "IsHoliday"]
ACCENT = "#2563eb"

# ----------------------------------------------------------------------------
# Page config + light custom styling
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Rossmann Sales Forecasting", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background: #f7f9fc; }
    h1, h2, h3 { color: #0f172a; font-family: 'Inter', system-ui, sans-serif; }
    .metric-card {
        background: #ffffff; border-radius: 16px; padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(15,23,42,.08); border: 1px solid #eef2f7;
    }
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
                f'<div class="value">{value}</div>{sub_html}</div>',
                unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading data…")
def load_data():
    if not (os.path.exists("train.csv") and os.path.exists("store.csv")):
        subprocess.run(["python", "generate_data.py"], check=True)
    store = pd.read_csv("store.csv")
    train = pd.read_csv("train.csv", parse_dates=["Date"])
    return store, train


@st.cache_data(show_spinner="Aggregating daily series…")
def build_daily(train):
    g = train.groupby("Date")
    df = pd.DataFrame({
        "Sales":      g.apply(lambda d: d.loc[d.Open == 1, "Sales"].mean()),
        "PromoShare": g["Promo"].mean(),
        "IsHoliday":  g["StateHoliday"].apply(lambda x: (x != "0").any()).astype(int),
    }).reset_index().sort_values("Date").reset_index(drop=True)
    df["Sales"] = df["Sales"].interpolate().bfill().ffill()
    df["DayOfWeek"] = df.Date.dt.dayofweek
    df["Month"] = df.Date.dt.month
    df["dow_sin"] = np.sin(2 * np.pi * df.DayOfWeek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df.DayOfWeek / 7)
    df["m_sin"] = np.sin(2 * np.pi * df.Month / 12)
    df["m_cos"] = np.cos(2 * np.pi * df.Month / 12)
    return df


def make_sequences(arr, seq_len):
    X, y = [], []
    for i in range(len(arr) - seq_len):
        X.append(arr[i:i + seq_len])
        y.append(arr[i + seq_len, 0])
    return np.array(X), np.array(y)


# ----------------------------------------------------------------------------
# Models (cached so they don't retrain on every interaction)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_nn(_df_values, kind, seq_len, epochs, split):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, SimpleRNN, LSTM, Dense, Dropout
    tf.keras.backend.clear_session(); tf.random.set_seed(SEED); np.random.seed(SEED)

    scaler = MinMaxScaler().fit(_df_values[:split])
    scaled = scaler.transform(_df_values)
    X, y = make_sequences(scaled, seq_len)
    tr = split - seq_len
    X_tr, X_te, y_tr, y_te = X[:tr], X[tr:], y[:tr], y[tr:]

    cell = SimpleRNN if kind == "SimpleRNN" else LSTM
    model = Sequential([Input((seq_len, _df_values.shape[1])),
                        cell(64), Dropout(0.15), Dense(1)])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    hist = model.fit(X_tr, y_tr, epochs=epochs, batch_size=32,
                     validation_split=0.15, verbose=0)

    s_min, s_max = scaler.data_min_[0], scaler.data_max_[0]
    inv = lambda s: np.asarray(s).ravel() * (s_max - s_min) + s_min
    pred, actual = inv(model.predict(X_te, verbose=0)), inv(y_te)
    metrics = _score(actual, pred)
    return model.get_weights(), scaler, pred, actual, metrics, hist.history


@st.cache_resource(show_spinner=False)
def train_sarima(_series_values, split):
    import statsmodels.api as sm
    s = pd.Series(_series_values)
    tr, te = s.iloc[:split], s.iloc[split:]
    model = sm.tsa.statespace.SARIMAX(tr, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7),
                                      enforce_stationarity=False,
                                      enforce_invertibility=False).fit(disp=False)
    fc = model.forecast(len(te)).values
    return fc, te.values, _score(te.values, fc)


def _score(actual, pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
        "MAE":  float(mean_absolute_error(actual, pred)),
        "MAPE": float(np.mean(np.abs((actual - pred) / actual)) * 100),
    }


def recursive_forecast(weights, scaler, df, seq_len, horizon, promo_share, kind):
    """Roll the NN forward `horizon` days, holding promo intensity at promo_share."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, SimpleRNN, LSTM, Dense, Dropout
    tf.keras.backend.clear_session()
    cell = SimpleRNN if kind == "SimpleRNN" else LSTM
    model = Sequential([Input((seq_len, len(FEATURES))), cell(64), Dropout(0.15), Dense(1)])
    model.set_weights(weights)

    scaled = scaler.transform(df[FEATURES].values.astype("float32"))
    window = list(scaled[-seq_len:])
    last_date = df.Date.iloc[-1]
    future_dates, preds_scaled = [], []
    s_min, s_max = scaler.data_min_[0], scaler.data_max_[0]
    for h in range(1, horizon + 1):
        d = last_date + pd.Timedelta(days=h)
        dow, mth = d.dayofweek, d.month
        feat = np.array([0,
                         np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7),
                         np.sin(2*np.pi*mth/12), np.cos(2*np.pi*mth/12),
                         promo_share, 0], dtype="float32")
        # scale the exogenous features using the fitted scaler ranges
        feat_scaled = (feat - scaler.data_min_) / (scaler.data_max_ - scaler.data_min_ + 1e-9)
        x = np.array(window[-seq_len:]).reshape(1, seq_len, len(FEATURES))
        p = float(model.predict(x, verbose=0).ravel()[0])
        new_row = feat_scaled.copy(); new_row[0] = p
        window.append(new_row)
        preds_scaled.append(p); future_dates.append(d)
    preds = np.array(preds_scaled) * (s_max - s_min) + s_min
    return pd.DataFrame({"Date": future_dates, "Forecast": preds})


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
store, train = load_data()
df = build_daily(train)
split = int(len(df) * 0.8)

with st.sidebar:
    st.title("📈 Sales Forecasting")
    st.caption("Rossmann Store Sales — RNN · LSTM · SARIMA")
    st.markdown("---")
    st.subheader("⚙️ Model settings")
    seq_len = st.slider("Look-back window (days)", 7, 35, 21, 7,
                        help="How many past days each prediction is based on.")
    epochs = st.slider("Training epochs", 20, 200, 80, 20,
                       help="More epochs = better fit but slower training.")
    st.markdown("---")
    n_stores = store.Store.nunique()
    st.metric("Stores in dataset", f"{n_stores}")
    st.metric("Days of history", f"{len(df)}")
    using_real = os.path.exists("train.csv") and train.shape[0] > 200000
    st.caption("✅ Real Kaggle data" if using_real else "🧪 Demo data (same schema as Kaggle)")

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
    fig = px.line(df, x="Date", y="Sales", color_discrete_sequence=[ACCENT])
    fig.update_layout(height=330, margin=dict(t=10, b=10), yaxis_title="Avg sales")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("A gentle upward trend, a yearly cycle peaking each December, and a "
               "strong weekly oscillation — the three signals the models must learn.")

    cA, cB = st.columns(2)
    with cA:
        st.markdown("### Weekly seasonality")
        names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow = open_df.assign(d=open_df.Date.dt.dayofweek).groupby("d").Sales.mean()
        fig = px.bar(x=names, y=dow.values, color=dow.values,
                     color_continuous_scale="Blues")
        fig.update_layout(height=300, margin=dict(t=10, b=10), showlegend=False,
                          coloraxis_showscale=False, xaxis_title="", yaxis_title="Avg sales")
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
    fig = px.bar(byt, x="StoreType", y="Sales", color="Sales",
                 color_continuous_scale="Viridis")
    fig.update_layout(height=300, margin=dict(t=10, b=10), coloraxis_showscale=False,
                      yaxis_title="Avg sales")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 2 — Model comparison
# ----------------------------------------------------------------------------
with tab2:
    st.markdown("### Train and compare the three models")
    st.caption("Neural nets and SARIMA are scored on the same held-out 20% test "
               "period (the most recent days). Click below to train.")
    go_train = st.button("🚀 Train & evaluate all models", type="primary")

    if go_train or st.session_state.get("trained"):
        st.session_state["trained"] = True
        vals = df[FEATURES].values.astype("float32")
        with st.spinner("Training SimpleRNN…"):
            w_rnn, sc_rnn, p_rnn, actual, m_rnn, h_rnn = train_nn(vals, "SimpleRNN", seq_len, epochs, split)
        with st.spinner("Training LSTM…"):
            w_lstm, sc_lstm, p_lstm, _, m_lstm, h_lstm = train_nn(vals, "LSTM", seq_len, epochs, split)
        with st.spinner("Fitting SARIMA…"):
            sar_fc, sar_actual, m_sar = train_sarima(df["Sales"].values.astype("float32"), split)

        # stash the best NN for the forecast tab
        st.session_state.update(dict(w_rnn=w_rnn, sc_rnn=sc_rnn, w_lstm=w_lstm,
                                     sc_lstm=sc_lstm, seq_len=seq_len))

        results = pd.DataFrame([
            {"Model": "SARIMA(1,1,1)(1,1,1)₇", **m_sar},
            {"Model": "SimpleRNN", **m_rnn},
            {"Model": "LSTM", **m_lstm},
        ]).sort_values("RMSE").reset_index(drop=True)
        best = results.iloc[0]["Model"]

        cols = st.columns(3)
        for col, (_, r) in zip(cols, results.iterrows()):
            with col:
                card(r["Model"], f"{r['MAPE']:.2f}%",
                     f"RMSE {r['RMSE']:,.0f} · MAE {r['MAE']:,.0f}",
                     winner=(r["Model"] == best))
        st.success(f"🏆 Best model: **{best}** — lowest error on the held-out period.")

        st.markdown("### Predicted vs actual — test period")
        test_dates = df["Date"].iloc[split:].values
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_dates, y=actual, name="Actual",
                                 line=dict(color="#0f172a", width=3)))
        fig.add_trace(go.Scatter(x=test_dates, y=p_rnn, name="SimpleRNN",
                                 line=dict(color="#2563eb", width=1.6)))
        fig.add_trace(go.Scatter(x=test_dates, y=p_lstm, name="LSTM",
                                 line=dict(color="#16a34a", width=1.6)))
        fig.add_trace(go.Scatter(x=df["Date"].iloc[split:].values, y=sar_fc, name="SARIMA",
                                 line=dict(color="#f59e0b", width=1.6, dash="dash")))
        fig.update_layout(height=420, margin=dict(t=10), yaxis_title="Avg sales",
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📉 Training curves (neural networks)"):
            cc1, cc2 = st.columns(2)
            for cc, h, name in [(cc1, h_rnn, "SimpleRNN"), (cc2, h_lstm, "LSTM")]:
                with cc:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=h["loss"], name="train"))
                    fig.add_trace(go.Scatter(y=h["val_loss"], name="validation"))
                    fig.update_layout(title=name, height=280, margin=dict(t=30),
                                      xaxis_title="epoch", yaxis_title="MSE (scaled)")
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👆 Click **Train & evaluate all models** to run the comparison.")

# ----------------------------------------------------------------------------
# TAB 3 — Forecast & simulate
# ----------------------------------------------------------------------------
with tab3:
    st.markdown("### Forecast future sales — and simulate a promotion")
    if not st.session_state.get("trained"):
        st.warning("Train the models first on the **Model Comparison** tab.")
    else:
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            kind = st.selectbox("Model", ["SimpleRNN", "LSTM"])
        with c2:
            horizon = st.slider("Forecast horizon (days)", 7, 60, 30, 7)
        with c3:
            promo = st.slider("Promotion scenario — share of stores on promo",
                              0.0, 1.0, round(float(train.Promo.mean()), 2), 0.01,
                              help="What-if lever. The neural-net forecast is computed at the "
                                   "historical promo level; this scenario then applies the "
                                   "empirically measured promotional uplift.")
        weights = st.session_state["w_rnn"] if kind == "SimpleRNN" else st.session_state["w_lstm"]
        scaler = st.session_state["sc_rnn"] if kind == "SimpleRNN" else st.session_state["sc_lstm"]

        # NN forecast at the in-distribution (historical mean) promo share — reliable
        p0 = float(train.Promo.mean())
        lift = train[train.Open == 1].groupby("Promo").Sales.mean()
        measured_uplift = lift[1] / lift[0] - 1
        expected = recursive_forecast(weights, scaler, df, st.session_state["seq_len"],
                                      horizon, p0, kind)
        factor = (1 + measured_uplift * promo) / (1 + measured_uplift * p0)
        fc = expected.assign(Forecast=expected.Forecast * factor)

        k1, k2, k3 = st.columns(3)
        with k1: card("Forecast total", f"€{fc.Forecast.sum():,.0f}", f"next {horizon} days")
        with k2: card("Daily average", f"€{fc.Forecast.mean():,.0f}")
        uplift = (factor - 1) * 100
        with k3: card("Scenario vs typical", f"{uplift:+.1f}%",
                      f"promo share {promo:.0%} vs {p0:.0%}")

        hist_tail = df.tail(90)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_tail.Date, y=hist_tail.Sales, name="History",
                                 line=dict(color="#94a3b8")))
        fig.add_trace(go.Scatter(x=expected.Date, y=expected.Forecast, name="Expected (typical promo)",
                                 line=dict(color="#94a3b8", width=1.6, dash="dot")))
        fig.add_trace(go.Scatter(x=fc.Date, y=fc.Forecast, name=f"{kind} — scenario",
                                 line=dict(color=ACCENT, width=2.5)))
        fig.add_vline(x=df.Date.iloc[-1], line_dash="dot", line_color="#cbd5e1")
        fig.update_layout(height=420, margin=dict(t=10), yaxis_title="Avg sales",
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"The neural network forecasts sales one day at a time (recursive "
                   f"forecasting) at the historical promotion level. The **scenario** line "
                   f"then applies the measured **+{measured_uplift*100:.1f}%** per-store "
                   f"promotional uplift to show the marketing what-if. Slide the promo share "
                   f"to compare.")

        st.download_button("⬇️ Download forecast (CSV)",
                           fc.to_csv(index=False).encode(),
                           file_name="sales_forecast.csv", mime="text/csv")
