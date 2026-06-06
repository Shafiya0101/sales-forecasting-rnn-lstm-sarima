"""
Run the full RNN/LSTM/SARIMA pipeline ONCE and save everything the app needs
into results.json. The deployed Streamlit app then just loads this file, so it
no longer needs TensorFlow or statsmodels at run time — which makes it deploy
cleanly on Streamlit Community Cloud's free tier.

Run locally:  python precompute.py
"""
import json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
import os, subprocess

SEED = 7
FEATURES = ["Sales", "dow_sin", "dow_cos", "m_sin", "m_cos", "PromoShare", "IsHoliday"]
SEQ_LEN, EPOCHS, HORIZON = 21, 150, 60

if not (os.path.exists("train.csv") and os.path.exists("store.csv")):
    subprocess.run(["python", "generate_data.py"], check=True)
train = pd.read_csv("train.csv", parse_dates=["Date"])

# ---- daily series + features ----
g = train.groupby("Date")
df = pd.DataFrame({
    "Sales":      g.apply(lambda d: d.loc[d.Open == 1, "Sales"].mean()),
    "PromoShare": g["Promo"].mean(),
    "IsHoliday":  g["StateHoliday"].apply(lambda x: (x != "0").any()).astype(int),
}).reset_index().sort_values("Date").reset_index(drop=True)
df["Sales"] = df["Sales"].interpolate().bfill().ffill()
df["DayOfWeek"] = df.Date.dt.dayofweek; df["Month"] = df.Date.dt.month
df["dow_sin"] = np.sin(2*np.pi*df.DayOfWeek/7); df["dow_cos"] = np.cos(2*np.pi*df.DayOfWeek/7)
df["m_sin"] = np.sin(2*np.pi*df.Month/12);     df["m_cos"] = np.cos(2*np.pi*df.Month/12)

vals = df[FEATURES].values.astype("float32")
SPLIT = int(len(vals) * 0.8)
scaler = MinMaxScaler().fit(vals[:SPLIT]); scaled = scaler.transform(vals)
s_min, s_max = scaler.data_min_[0], scaler.data_max_[0]
inv = lambda s: np.asarray(s).ravel() * (s_max - s_min) + s_min

def make_seq(a, s):
    X, y = [], []
    for i in range(len(a) - s): X.append(a[i:i+s]); y.append(a[i+s, 0])
    return np.array(X), np.array(y)
X, y = make_seq(scaled, SEQ_LEN)
tr = SPLIT - SEQ_LEN
X_tr, X_te, y_tr, y_te = X[:tr], X[tr:], y[:tr], y[tr:]
actual = inv(y_te)

def score(a, p):
    return {"RMSE": float(np.sqrt(mean_squared_error(a, p))),
            "MAE": float(mean_absolute_error(a, p)),
            "MAPE": float(np.mean(np.abs((a - p) / a)) * 100)}

def build(kind):
    tf.keras.backend.clear_session(); tf.random.set_seed(SEED); np.random.seed(SEED)
    cell = SimpleRNN if kind == "rnn" else LSTM
    m = Sequential([Input((SEQ_LEN, X.shape[2])), cell(64), Dropout(0.15), Dense(1)])
    m.compile(tf.keras.optimizers.Adam(1e-3), "mse"); return m

def recursive_forecast(model, promo_share):
    window = list(scaled[-SEQ_LEN:]); last = df.Date.iloc[-1]; preds = []
    for h in range(1, HORIZON + 1):
        d = last + pd.Timedelta(days=h); dow, mth = d.dayofweek, d.month
        feat = np.array([0, np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7),
                         np.sin(2*np.pi*mth/12), np.cos(2*np.pi*mth/12), promo_share, 0], "float32")
        fsd = (feat - scaler.data_min_) / (scaler.data_max_ - scaler.data_min_ + 1e-9)
        x = np.array(window[-SEQ_LEN:]).reshape(1, SEQ_LEN, len(FEATURES))
        p = float(model.predict(x, verbose=0).ravel()[0])
        nr = fsd.copy(); nr[0] = p; window.append(nr); preds.append(p)
    return (np.array(preds) * (s_max - s_min) + s_min).tolist()

p0 = float(train.Promo.mean())
out = {"models": {}, "forecasts": {}, "curves": {}}

for kind, name in [("rnn", "SimpleRNN"), ("lstm", "LSTM")]:
    print("training", name, "…")
    m = build(kind)
    h = m.fit(X_tr, y_tr, epochs=EPOCHS, batch_size=32, validation_split=0.15, verbose=0)
    pred = inv(m.predict(X_te, verbose=0))
    out["models"][name] = {"pred": pred.tolist(), "metrics": score(actual, pred)}
    out["curves"][name] = {"loss": [float(v) for v in h.history["loss"]],
                           "val_loss": [float(v) for v in h.history["val_loss"]]}
    out["forecasts"][name] = recursive_forecast(m, p0)

print("fitting SARIMA …")
ts = df.set_index("Date")["Sales"].asfreq("D")
sar = sm.tsa.statespace.SARIMAX(ts.iloc[:SPLIT], order=(1,1,1), seasonal_order=(1,1,1,7),
        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
sar_fc = sar.forecast(len(ts) - SPLIT).values
out["models"]["SARIMA"] = {"pred": sar_fc.tolist(), "metrics": score(ts.iloc[SPLIT:].values, sar_fc)}

# seasonal-naive baseline RMSE
sn = ts.shift(7).iloc[SPLIT:].values
out["baseline_rmse"] = float(np.sqrt(mean_squared_error(ts.iloc[SPLIT:].values, sn)))

# shared metadata for the app
lift = train[train.Open == 1].groupby("Promo").Sales.mean()
out["meta"] = {
    "split": int(SPLIT), "seq_len": SEQ_LEN, "horizon": HORIZON,
    "p0": p0, "measured_uplift": float(lift[1] / lift[0] - 1),
    "test_dates": df["Date"].iloc[SPLIT:].dt.strftime("%Y-%m-%d").tolist(),
    "actual": actual.tolist(),
    "forecast_dates": [(df.Date.iloc[-1] + pd.Timedelta(days=h)).strftime("%Y-%m-%d")
                       for h in range(1, HORIZON + 1)],
    "hist_tail_dates": df["Date"].tail(90).dt.strftime("%Y-%m-%d").tolist(),
    "hist_tail_sales": df["Sales"].tail(90).round(1).tolist(),
}
json.dump(out, open("results.json", "w"))
print("Saved results.json")
for nm, d in out["models"].items():
    print(f"  {nm:10s} RMSE={d['metrics']['RMSE']:,.0f} MAPE={d['metrics']['MAPE']:.2f}%")
