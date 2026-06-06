"""
Generate a synthetic dataset that matches the EXACT schema of the Kaggle
Rossmann Store Sales competition (store.csv + train.csv).

The point is realism of *structure and behaviour*, so the notebook and the
Streamlit app run end-to-end here. When you download the real data from
    https://www.kaggle.com/competitions/rossmann-store-sales/data
you can drop store.csv and train.csv in the same folder and every cell works
unchanged, because the column names and dtypes are identical.

Real Rossmann columns
---------------------
train.csv : Store, DayOfWeek, Date, Sales, Customers, Open, Promo,
            StateHoliday, SchoolHoliday
store.csv : Store, StoreType, Assortment, CompetitionDistance,
            CompetitionOpenSinceMonth, CompetitionOpenSinceYear, Promo2,
            Promo2SinceWeek, Promo2SinceYear, PromoInterval
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N_STORES = 50                       # real set has 1115; 50 keeps it fast & realistic
START = pd.Timestamp("2013-01-01")
END = pd.Timestamp("2015-07-31")    # same window as the real competition
dates = pd.date_range(START, END, freq="D")

# ----------------------------------------------------------------------
# store.csv
# ----------------------------------------------------------------------
store_types = rng.choice(list("abcd"), size=N_STORES, p=[0.55, 0.15, 0.20, 0.10])
assortment = rng.choice(list("abc"), size=N_STORES, p=[0.5, 0.1, 0.4])
comp_dist = rng.integers(20, 20000, size=N_STORES).astype(float)
comp_dist[rng.random(N_STORES) < 0.03] = np.nan  # a few missing, like the real data
promo2 = rng.integers(0, 2, size=N_STORES)

store = pd.DataFrame({
    "Store": np.arange(1, N_STORES + 1),
    "StoreType": store_types,
    "Assortment": assortment,
    "CompetitionDistance": comp_dist,
    "CompetitionOpenSinceMonth": rng.integers(1, 13, size=N_STORES),
    "CompetitionOpenSinceYear": rng.integers(2000, 2015, size=N_STORES),
    "Promo2": promo2,
    "Promo2SinceWeek": np.where(promo2 == 1, rng.integers(1, 52, N_STORES), np.nan),
    "Promo2SinceYear": np.where(promo2 == 1, rng.integers(2010, 2015, N_STORES), np.nan),
    "PromoInterval": [
        rng.choice(["Jan,Apr,Jul,Oct", "Feb,May,Aug,Nov", "Mar,Jun,Sept,Dec"])
        if p == 1 else np.nan for p in promo2
    ],
})

# per-store baseline sales level (store type a busiest)
type_level = {"a": 1.0, "b": 1.45, "c": 0.85, "d": 0.95}
base_sales = np.array([6000 * type_level[t] for t in store_types]) * rng.uniform(0.8, 1.2, N_STORES)

# ----------------------------------------------------------------------
# train.csv  (one row per store per day)
# ----------------------------------------------------------------------
# German state / school holidays (approximate, just to create realistic flags)
state_holidays = pd.to_datetime([
    "2013-01-01", "2013-03-29", "2013-04-01", "2013-05-01", "2013-12-25", "2013-12-26",
    "2014-01-01", "2014-04-18", "2014-04-21", "2014-05-01", "2014-12-25", "2014-12-26",
    "2015-01-01", "2015-04-03", "2015-04-06", "2015-05-01",
])
school_holiday_months = {1, 4, 7, 8, 12}  # rough school-break months

records = []
for s in range(1, N_STORES + 1):
    lvl = base_sales[s - 1]
    for d in dates:
        dow = d.dayofweek + 1               # 1=Mon ... 7=Sun  (matches Rossmann)
        # Sundays: ~90% of stores closed; some closed random days
        open_flag = 0 if (dow == 7 and rng.random() < 0.45) else int(rng.random() > 0.02)
        promo = int(rng.random() < 0.38)    # ~38% of open days on promo
        is_state = d in state_holidays
        state_hol = rng.choice(["a", "b", "c"]) if is_state else "0"
        school_hol = int(d.month in school_holiday_months and rng.random() < 0.4)

        if open_flag == 0:
            sales, customers = 0, 0
        else:
            # ---- multiplicative seasonal / behavioural structure ----
            week = {1: 1.10, 2: 1.02, 3: 1.00, 4: 1.00, 5: 1.05, 6: 0.78, 7: 0.55}[dow]
            # yearly seasonality + strong December spike
            doy = d.dayofyear
            yearly = 1 + 0.12 * np.sin(2 * np.pi * (doy - 80) / 365)
            december = 1.45 if d.month == 12 and d.day >= 5 else 1.0
            promo_lift = 1.28 if promo else 1.0
            hol_effect = 0.6 if is_state else 1.0
            trend = 1 + 0.04 * ((d - START).days / 365)   # slight upward trend
            noise = rng.normal(1.0, 0.10)
            sales = lvl * week * yearly * december * promo_lift * hol_effect * trend * noise
            sales = max(0, int(sales))
            customers = max(0, int(sales / rng.uniform(8.5, 11.0)))

        records.append((s, dow, d, sales, customers, open_flag, promo, state_hol, school_hol))

train = pd.DataFrame(records, columns=[
    "Store", "DayOfWeek", "Date", "Sales", "Customers", "Open", "Promo",
    "StateHoliday", "SchoolHoliday"])
train["Date"] = train["Date"].dt.strftime("%Y-%m-%d")

store.to_csv("store.csv", index=False)
train.to_csv("train.csv", index=False)
print(f"store.csv  -> {store.shape}")
print(f"train.csv  -> {train.shape}")
print(train.head())
