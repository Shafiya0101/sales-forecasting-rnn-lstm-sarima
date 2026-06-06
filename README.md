# Rossmann Sales Forecasting — RNN · LSTM · SARIMA

A complete, exam-ready sales-forecasting project on the **Rossmann Store Sales**
dataset. It compares two recurrent neural networks (SimpleRNN, LSTM) against a
classical seasonal **SARIMA** model, and ships with an interactive **Streamlit**
dashboard.

## What's inside

| File | Description |
|---|---|
| `Sales_Forecasting_RNN_LSTM_SARIMA.ipynb` | The main deliverable — full pipeline with EDA, three models, evaluation and discussion. Already executed (graphs + outputs embedded). |
| `app.py` | Streamlit dashboard: EDA, live model training/comparison, and an interactive forecast with a promotion what-if. |
| `generate_data.py` | Creates a realistic stand-in dataset with the **exact Rossmann schema** so everything runs without the Kaggle download. |
| `requirements.txt` | Pinned, tested dependencies. |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

## Using the real Kaggle data (recommended for submission)

1. Download `train.csv` and `store.csv` from
   <https://www.kaggle.com/competitions/rossmann-store-sales/data>
   (free Kaggle account + accept the competition rules).
2. Put both files in this folder.
3. Run — the code detects them automatically. **No code changes needed**; the
   synthetic generator uses identical column names and types.

If the two CSVs are absent, `generate_data.py` runs automatically and produces a
realistic synthetic dataset so the notebook and app still work end-to-end.

## Run the notebook

```bash
jupyter notebook Sales_Forecasting_RNN_LSTM_SARIMA.ipynb
```

It is already executed, so you can also just read it. To re-run: *Kernel →
Restart & Run All*.

## Run the app

```bash
streamlit run app.py
```

Then open the URL it prints (default <http://localhost:8501>). Three tabs:

1. **Overview & EDA** — KPIs and the key sales patterns.
2. **Model Comparison** — click *Train & evaluate all models* to train the RNN,
   LSTM and SARIMA and score them on a held-out test period.
3. **Forecast & Simulate** — forecast future days with the chosen network and use
   the promotion slider for a marketing what-if.

## Method in one paragraph

We aggregate the data to one clean series — **average sales per open store per
day** — so all three models forecast the same target. Features are scaled with a
`MinMaxScaler` **fit on the training portion only** (no leakage), and the split is
**chronological** (the test set is the most recent 20%). The neural nets read a
21-day window and predict the next day; SARIMA uses order `(1,1,1)(1,1,1)₇` to
capture the weekly cycle.

## Results (held-out test period)

| Model | RMSE | MAPE |
|---|---|---|
| SimpleRNN | ~505 | ~4.9% |
| LSTM | ~519 | ~4.7% |
| SARIMA | ~1431 | ~17.3% |

Both recurrent networks cut the error of the classical model by roughly
two-thirds. SimpleRNN and LSTM are comparable here — on a short, smooth daily
series the LSTM's long-memory gating adds little, a clean illustration that model
complexity should match data complexity. (Exact numbers vary slightly by hardware
but the ranking is stable; seeds are fixed at 7.)
