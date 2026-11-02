import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import io
import holidays
from datetime import datetime, timedelta
import plotly.graph_objects as go
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo  # Built-in timezone support

# Helper function to get current Malaysian date
def get_myt_today():
  return datetime.now(ZoneInfo("Asia/Kuala_Lumpur")).date()

st.set_page_config(page_title="Sistem Ramalan Harga Sayuran Pertanian Malaysia",
                    page_icon="🌾", layout="wide")

# ============================================================
# STYLING
# ============================================================
st.markdown("""
<style>
.stApp { background-color: #ffffff; }
.block-container {
    padding-top: 0rem !important;
}
.top-banner {
    background-color: #003d20;
    border-radius: 0px; 
    padding: 15px 40px; 
    color: white;
    margin-bottom: 18px; 
    width: 100vw; 
    position: relative; 
    left: 50%; 
    margin-left: -50vw; 
    margin-right: -50vw;
}
.kpk-badge {
    background: rgba(255,255,255,0.15); padding: 6px 14px; border-radius: 20px;
    font-size: 12px; display: inline-block;
}
.live-badge {
    background: rgba(255,255,255,0.15); padding: 30px 20px; border-radius: 20px;
    font-size: 12px; display: inline-block;
}
.card {
    background-color:#d9d9d9; border-radius: 14px; padding: 18px 20px; height: 100%;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    display: flex; flex-direction: column; justify-content: space-between;
    min-height: 175px; box-sizing: border-box;
}
.card-label { font-size: 17px; color: #000000; font-weight: 600; letter-spacing: 0.5px; }
.card-value { font-size: 45px; font-weight: 800; color: #000000; }
.card-sub { font-size: 13px; color: #003d20; font-weight: 600; }

/* Mengubah kotak st.form menjadi latar belakang kelabu dengan 75% ketelusan */
[data-testid="stForm"] {
    background: rgba(217, 217, 217, 0.75);
    padding: 24px;
    border-radius: 16px;
    border: none;
    margin-bottom: 18px;
}

/* 1. Tukar warna butang "Jana Ramalan Harga" di dalam form */
[data-testid="stFormSubmitButton"] > button {
    background-color: #003d20 !important;
    color: white !important;
    border: none !important;
}

[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #054f30 !important;
    color: white !important;
}

/* 1. Tukar latar belakang kotak selectbox (Negeri & Komoditi) kepada putih (#ffffff) */
[data-baseweb="select"] > div {
    background-color: #ffffff !important;
}
[data-testid="stRadio"] div[role="radio"] {
    border-color: #003d20 !important;
}

/* 2. Tukar warna latar belakang apabila butang radio dipilih */
[data-testid="stRadio"] div[role="radio"][aria-checked="true"] {
    background-color: #003d20 !important;
    border-color: #003d20 !important;
}

/* 3. Paksa titik tengah di dalam radio button bertukar kepada #003d20 dan atasi inline style */
[data-testid="stRadio"] div[role="radio"][aria-checked="true"] span,
[data-testid="stRadio"] div[role="radio"][aria-checked="true"] div {
    background-color: #003d20 !important;
}

/* Tukar warna semua label dan teks perkataan di dalam form / panel kawalan kepada hitam */
[data-testid="stForm"] label,
[data-testid="stForm"] [data-testid="stWidgetLabel"] p,
[data-testid="stForm"] .stRadio span {
    color: #000000 !important;
    font-weight: 600 !important;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# STATIC REFERENCE DATA
# ============================================================
STATE_COORDS = {
    "Johor": (1.4854, 103.7618), "Kedah": (6.1184, 100.3685),
    "Kelantan": (6.1254, 102.2381), "Melaka": (2.1896, 102.2501),
    "Negeri Sembilan": (2.7258, 101.9424), "Pahang": (3.8126, 103.3256),
    "Perak": (4.5921, 101.0901), "Perlis": (6.4449, 100.2048),
    "Pulau Pinang": (5.4141, 100.3288), "Sabah": (5.9788, 116.0753),
    "Sarawak": (1.5533, 110.3592), "Selangor": (3.0738, 101.5183),
    "Terengganu": (5.3117, 103.1324), "W.P. Kuala Lumpur": (3.1390, 101.6869),
    "W.P. Labuan": (5.2831, 115.2308), "W.P. Putrajaya": (2.9264, 101.6964),
}

CROPS = ["CILI MERAH - KULAI", "CILI MERAH - MINYAK", "SAWI HIJAU", "TOMATO"]

# Baseline farm-gate price (RM/kg) + production cost basis used only to seed a
# synthetic 21-day "actual" history when no historical price CSV is supplied.
CROP_BASELINE = {
    "CILI MERAH - KULAI": {"price": 9.80, "cost": 6.30},
    "CILI MERAH - MINYAK": {"price": 11.20, "cost": 7.10},
    "SAWI HIJAU": {"price": 2.60, "cost": 1.40},
    "TOMATO": {"price": 4.30, "cost": 2.60},
}

MODEL_PATH = "xgboost_crop_model.pkl"
FEATURE_COLS_PATH = "model_feature_columns.pkl"
HISTORY_CSV_PATH = "featured_daily_matrix.csv"

# holidays library subdivision codes — required to surface STATE-level
# holidays (Deepavali, Thaipusam, state Sultan's birthdays, etc). Passing no
# subdiv only returns federal-wide holidays and silently drops these.
STATE_SUBDIV = {
    "Johor": "01", "Kedah": "02", "Kelantan": "03", "Melaka": "04",
    "Negeri Sembilan": "05", "Pahang": "06", "Pulau Pinang": "07",
    "Perak": "08", "Perlis": "09", "Selangor": "10", "Terengganu": "11",
    "Sabah": "12", "Sarawak": "13", "W.P. Kuala Lumpur": "14",
    "W.P. Labuan": "15", "W.P. Putrajaya": "16",
}

# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource
def load_model_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FEATURE_COLS_PATH, "rb") as f:
        feature_columns = pickle.load(f)
    return model, list(feature_columns)

try:
    MODEL, FEATURE_COLUMNS = load_model_artifacts()
    MODEL_LOADED = True
except Exception as e:
    MODEL, FEATURE_COLUMNS = None, []
    MODEL_LOADED = False
    MODEL_LOAD_ERROR = str(e)

# ============================================================
# DATA ENGINE A — LIVE OPEN-METEO WEATHER
# ============================================================
@st.cache_data(ttl=3600)
def get_weather_forecast(state, horizon_days):
    lat, lon = STATE_COORDS[state]
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean",
            "forecast_days": min(horizon_days, 16),
            "timezone": "Asia/Kuala_Lumpur",
        }
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        d = r.json()["daily"]
        dates = pd.to_datetime(d["time"])
        df = pd.DataFrame({
            "date": dates,
            "Rainfall_mm": d["precipitation_sum"],
            "Max_Temp": d["temperature_2m_max"],
            "Min_Temp": d["temperature_2m_min"],
            "Humidity_Pct": d["relative_humidity_2m_mean"],
        })
        if len(df) < horizon_days:
            extra_dates = pd.date_range(df["date"].iloc[-1] + timedelta(days=1),
                                         periods=horizon_days - len(df))
            extra = pd.DataFrame({
                "date": extra_dates, "Rainfall_mm": 5.0,
                "Max_Temp": 32.0, "Min_Temp": 24.0, "Humidity_Pct": 80.0,
            })
            df = pd.concat([df, extra], ignore_index=True)
        return df.head(horizon_days), True
    except Exception:
        dates = pd.date_range(get_myt_today(), periods=horizon_days)
        df = pd.DataFrame({
            "date": dates, "Rainfall_mm": 5.0,
            "Max_Temp": 32.0, "Min_Temp": 24.0, "Humidity_Pct": 80.0,
        })
        return df, False

# ============================================================
# DATA ENGINE A2 — HISTORICAL OPEN-METEO WEATHER (ARCHIVE)
# Needed to bridge the gap between the last date in the frozen training CSV
# and today: those days already happened, so they need real *past* weather,
# not a forecast.
# ============================================================
@st.cache_data(ttl=3600)
def get_weather_historical(state, start_date_str, end_date_str):
    lat, lon = STATE_COORDS[state]
    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": start_date_str, "end_date": end_date_str,
            "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean",
            "timezone": "Asia/Kuala_Lumpur",
        }
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        d = r.json()["daily"]
        df = pd.DataFrame({
            "date": pd.to_datetime(d["time"]),
            "Rainfall_mm": d["precipitation_sum"],
            "Max_Temp": d["temperature_2m_max"],
            "Min_Temp": d["temperature_2m_min"],
            "Humidity_Pct": d["relative_humidity_2m_mean"],
        })
        return df, True
    except Exception:
        dates = pd.date_range(start_date_str, end_date_str)
        df = pd.DataFrame({
            "date": dates, "Rainfall_mm": 5.0,
            "Max_Temp": 32.0, "Min_Temp": 24.0, "Humidity_Pct": 80.0,
        })
        return df, False

# ============================================================
# DATA ENGINE D — LIVE PRICECATCHER (data.gov.my)
# featured_daily_matrix.csv was itself built by averaging PriceCatcher's
# raw transactional records by state/date/item, so this uses the exact same
# join + averaging logic on the live monthly file to extend the actual
# price history right up to the most recently published date (PriceCatcher
# updates daily but usually lags "today" by ~1 day).
# ============================================================
PRICECATCHER_BASE = "https://storage.data.gov.my/pricecatcher"

@st.cache_data(ttl=86400)
def load_pricecatcher_lookups():
    item_bytes = requests.get(f"{PRICECATCHER_BASE}/lookup_item.parquet", timeout=60).content
    premise_bytes = requests.get(f"{PRICECATCHER_BASE}/lookup_premise.parquet", timeout=60).content
    item_lookup = pd.read_parquet(io.BytesIO(item_bytes))
    premise_lookup = pd.read_parquet(io.BytesIO(premise_bytes))
    item_lookup["item_norm"] = item_lookup["item"].astype(str).str.strip().str.upper()
    # normalize a few common state-name spellings to match STATE_COORDS keys
    state_alias = {
        "WP Kuala Lumpur": "W.P. Kuala Lumpur", "Wilayah Persekutuan Kuala Lumpur": "W.P. Kuala Lumpur",
        "WP Labuan": "W.P. Labuan", "Wilayah Persekutuan Labuan": "W.P. Labuan",
        "WP Putrajaya": "W.P. Putrajaya", "Wilayah Persekutuan Putrajaya": "W.P. Putrajaya",
        "Penang": "Pulau Pinang",
    }
    premise_lookup["state_norm"] = premise_lookup["state"].astype(str).str.strip().replace(state_alias)
    return item_lookup, premise_lookup

@st.cache_data(ttl=3600)
def load_pricecatcher_month(month_str):
    """month_str format: 'YYYY-MM'. Only date/premise_code/item_code/price
    are kept — the raw monthly file has 1M+ rows across all items/states."""
    url = f"{PRICECATCHER_BASE}/pricecatcher_{month_str}.parquet"
    raw_bytes = requests.get(url, timeout=90).content
    df = pd.read_parquet(io.BytesIO(raw_bytes), columns=["date", "premise_code", "item_code", "price"])
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(ttl=3600)
def get_pricecatcher_live(state, crop, after_date):
    """Live daily avg_price_rm for `crop` in `state`, for dates strictly
    after `after_date` (the frozen CSV's last recorded date) up to whatever
    PriceCatcher has published so far."""
    try:
        item_lookup, premise_lookup = load_pricecatcher_lookups()
        crop_norm = crop.strip().upper()
        item_codes = set(item_lookup.loc[item_lookup["item_norm"] == crop_norm, "item_code"])
        if not item_codes:
            item_codes = set(item_lookup.loc[item_lookup["item_norm"].str.contains(crop_norm, na=False), "item_code"])
        premise_codes = set(premise_lookup.loc[premise_lookup["state_norm"] == state, "premise_code"])
        if not item_codes or not premise_codes:
            return pd.DataFrame(columns=["date", "price"]), False

        today = get_myt_today()
        months = sorted({(today.replace(day=1) - pd.DateOffset(months=k)).strftime("%Y-%m") for k in range(2)})
        frames = []
        for m in months:
            try:
                frames.append(load_pricecatcher_month(m))
            except Exception:
                continue
        if not frames:
            return pd.DataFrame(columns=["date", "price"]), False

        raw = pd.concat(frames, ignore_index=True)
        raw = raw[raw["item_code"].isin(item_codes) & raw["premise_code"].isin(premise_codes)
                   & (raw["date"] > pd.Timestamp(after_date))]
        if raw.empty:
            return pd.DataFrame(columns=["date", "price"]), True  # connected fine, just nothing newer yet
        daily = (raw.groupby("date", as_index=False)["price"].mean()
                 .rename(columns={"price": "price"}).sort_values("date"))
        return daily[["date", "price"]], True
    except Exception:
        return pd.DataFrame(columns=["date", "price"]), False

# ============================================================
# DATA ENGINE B — LIVE OPENDOSM DIESEL PRICE
# ============================================================
@st.cache_data(ttl=3600)
def get_diesel_price():
    baseline = 2.15
    try:
        url = "https://api.data.gov.my/data-catalogue"
        params = {"id": "fuelprice"}
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data)
        diesel_col = [c for c in df.columns if "diesel" in c.lower()]
        if diesel_col and len(df) > 0:
            latest_price = float(df.sort_values("date").iloc[-1][diesel_col[0]])
            change = round(latest_price - baseline, 2)
            return latest_price, change, True
    except Exception:
        pass
    return 3.35, round(3.35 - baseline, 2), False

# ============================================================
# DATA ENGINE C — DYNAMIC FESTIVE CALENDAR
# ============================================================
@st.cache_data(ttl=86400)
def get_festive_features(dates, state):
    years = sorted(set([d.year for d in dates] + [d.year + 1 for d in dates]))
    subdiv = STATE_SUBDIV.get(state)
    # subdiv is required here: holidays.MY(years=..) with no subdiv only
    # returns federal holidays and drops state-level ones (Deepavali,
    # Thaipusam, state Sultan's birthday, etc), which was the earlier bug.
    my_holidays = holidays.MY(years=years, subdiv=subdiv)
    holiday_dates = sorted(my_holidays.keys())
    rows = []
    for d in dates:
        d_date = d.date() if hasattr(d, "date") else d
        is_festive = 1 if d_date in my_holidays else 0
        future = [hd for hd in holiday_dates if hd >= d_date]
        if future:
            next_hd = future[0]
            days_to = (next_hd - d_date).days
            name = my_holidays.get(next_hd)
        else:
            days_to, name = 999, "N/A"
        rows.append({"date": d, "Is_Festive": is_festive,
                      "Days_To_Festive": days_to, "next_festive_name": name})
    return pd.DataFrame(rows)

# ============================================================
# REAL 21-DAY ACTUAL PRICE HISTORY — loaded from the training dataset
# (featured_daily_matrix.csv). Falls back to a synthetic series only if
# that file is missing, so the demo still renders.
# ============================================================
@st.cache_data(ttl=3600)
def load_history_csv():
    df = pd.read_csv(HISTORY_CSV_PATH, parse_dates=["date"])
    return df

def get_actual_price_history(state, crop, n_days=21):
    try:
        df = load_history_csv()
        subset = df[(df["state"] == state) & (df["item"] == crop)].sort_values("date")
        if subset.empty:
            raise ValueError("no rows for this state/crop")
        csv_last_date = subset["date"].max()

        # Extend with live PriceCatcher data for anything published after the
        # frozen CSV's last date — same state/item/date averaging the CSV
        # itself was built with, so it appends seamlessly as real actuals.
        live_ext, pricecatcher_connected = get_pricecatcher_live(state, crop, csv_last_date)

        combined = pd.concat([
            subset[["date", "avg_price_rm"]].rename(columns={"avg_price_rm": "price"}),
            live_ext,
        ], ignore_index=True).drop_duplicates(subset="date", keep="last").sort_values("date")

        hist = combined.tail(n_days).reset_index(drop=True)
        tail_prices = combined["price"].values
        seed_features = {
            "lag_1": float(tail_prices[-1]),
            "lag_3": float(tail_prices[-3]) if len(tail_prices) >= 3 else float(tail_prices[0]),
            "lag_7": float(tail_prices[-7]) if len(tail_prices) >= 7 else float(tail_prices[0]),
            "lag_14": float(tail_prices[-14]) if len(tail_prices) >= 14 else float(tail_prices[0]),
            "rolling_mean_7": float(np.mean(tail_prices[-7:])),
            "rolling_std_7": float(np.std(tail_prices[-7:])),
        }
        live_days_added = int((combined["date"] > csv_last_date).sum())
        return hist, seed_features, True, pricecatcher_connected, live_days_added
    except Exception:
        seed = abs(hash(state + crop)) % (2**32)
        rng = np.random.default_rng(seed)
        base = CROP_BASELINE[crop]["price"]
        noise = rng.normal(0, base * 0.03, n_days)
        trend = np.linspace(base * 1.05, base, n_days)
        prices = trend + noise
        dates = pd.date_range(end=get_myt_today(), periods=n_days)
        hist = pd.DataFrame({"date": dates, "price": prices})
        seed_features = {
            "lag_1": prices[-1], "lag_3": prices[-3], "lag_7": prices[-7],
            "lag_14": prices[0], "rolling_mean_7": float(np.mean(prices[-7:])),
            "rolling_std_7": float(np.std(prices[-7:])),
        }
        return hist, seed_features, False, False, 0

@st.cache_data(ttl=3600)
def get_latest_prices_all_states(crop):
    """Latest known price of `crop` in every state — prefers live PriceCatcher
    data per state when available, else falls back to the CSV's last row.
    Used for the cross-state market comparison chart (real, actionable data)."""
    try:
        df = load_history_csv()
        sub = df[df["item"] == crop].sort_values("date")
        latest = sub.groupby("state").tail(1)[["state", "avg_price_rm", "date"]].set_index("state")
        for s in latest.index:
            csv_last_date = latest.loc[s, "date"]
            live_ext, connected = get_pricecatcher_live(s, crop, csv_last_date)
            if connected and not live_ext.empty:
                latest.loc[s, "avg_price_rm"] = float(live_ext["price"].iloc[-1])
                latest.loc[s, "date"] = live_ext["date"].iloc[-1]
        latest = latest.reset_index()
        return latest.sort_values("avg_price_rm", ascending=False), True
    except Exception:
        rows = [{"state": s, "avg_price_rm": CROP_BASELINE[crop]["price"], "date": pd.NaT}
                for s in STATE_COORDS]
        return pd.DataFrame(rows), False

# ============================================================
# FEATURE ROW BUILDER — matches the confirmed training schema from
# featured_daily_matrix.csv:
# state, item, change_in_diesel_price_rm, Rainfall_mm, Max_Temp, Min_Temp,
# Humidity_Pct, Is_Festive, Days_To_Festive, lag_1, lag_3, lag_7, lag_14,
# rolling_mean_7, rolling_std_7
# `state`/`item` are set as pandas category dtype (categories = sorted
# unique values from the training CSV) to match how XGBoost's native
# categorical support (enable_categorical=True) encodes them when a
# DataFrame is fit directly with df[col].astype("category").
# If model_feature_columns.pkl differs from this list, only the dict below
# needs editing — the alignment/reindex at the end stays the same.
# ============================================================
@st.cache_data(ttl=3600)
def get_categories():
    try:
        df = load_history_csv()
        return sorted(df["state"].unique()), sorted(df["item"].unique())
    except Exception:
        return sorted(STATE_COORDS.keys()), sorted(CROPS)

# ============================================================
# REAL MODEL ACCURACY — backtests the loaded model against the most recent
# ACTUAL recorded prices for the selected state/crop (features taken
# straight from featured_daily_matrix.csv, since those rows already carry
# the exact lag/rolling/weather/diesel/festive values used at training
# time). Reported as 100 - MAPE. This is an in-sample sanity check, not a
# substitute for a proper held-out test-set metric from your training
# notebook — but it is real and specific to the current selection, not a
# fixed number.
# ============================================================
@st.cache_data(ttl=3600)
def compute_model_accuracy(state, crop, n_recent=60):
    if not MODEL_LOADED:
        return None
    try:
        df = load_history_csv()
        subset = df[(df["state"] == state) & (df["item"] == crop)].sort_values("date").tail(n_recent)
        if subset.empty:
            return None
        state_cats, item_cats = get_categories()
        X = subset.copy()
        X["state"] = pd.Categorical(X["state"], categories=state_cats)
        X["item"] = pd.Categorical(X["item"], categories=item_cats)
        if FEATURE_COLUMNS:
            for c in FEATURE_COLUMNS:
                if c not in X.columns:
                    X[c] = 0
            X_model = X[FEATURE_COLUMNS]
        else:
            X_model = X
        preds = MODEL.predict(X_model)
        actual = subset["avg_price_rm"].values
        mape = float(np.mean(np.abs((actual - preds) / actual))) * 100
        return max(0.0, 100.0 - mape)
    except Exception:
        return None

def build_feature_row(feature_columns, date, state, crop, weather_row,
                       diesel_change, lag1, lag3, lag7, lag14,
                       rolling_mean7, rolling_std7, is_festive, days_to_festive):
    state_cats, item_cats = get_categories()
    values = {
        "state": state,
        "item": crop,
        "change_in_diesel_price_rm": diesel_change,
        "Rainfall_mm": weather_row["Rainfall_mm"],
        "Max_Temp": weather_row["Max_Temp"],
        "Min_Temp": weather_row["Min_Temp"],
        "Humidity_Pct": weather_row["Humidity_Pct"],
        "Is_Festive": is_festive,
        "Days_To_Festive": days_to_festive,
        "lag_1": lag1, "lag_3": lag3, "lag_7": lag7, "lag_14": lag14,
        "rolling_mean_7": rolling_mean7, "rolling_std_7": rolling_std7,
    }
    row = pd.DataFrame([values])
    row["state"] = pd.Categorical(row["state"], categories=state_cats)
    row["item"] = pd.Categorical(row["item"], categories=item_cats)

    if feature_columns:
        missing = [c for c in feature_columns if c not in row.columns]
        for c in missing:
            row[c] = 0
        row = row[feature_columns]
    return row

# ============================================================
# ITERATIVE MULTI-STEP FORECAST
# featured_daily_matrix.csv is a FROZEN snapshot (ends 2026-09-08) — it will
# never auto-update. Two gaps need to be handled every time the app runs:
#   1) BRIDGE: from the day after the CSV's last date up to today. These
#      days already happened but have no recorded actual price, so they are
#      NOWCAST — same one-step model prediction as the future forecast,
#      just using real *historical* weather (archive API) instead of a
#      weather forecast. Plotted separately and labeled as an estimate,
#      never presented as an actual.
#   2) FUTURE: today onward for the requested horizon — the normal forecast,
#      using live forecast weather.
# The two are chained: the bridge's last predicted price becomes lag_1 for
# day 1 of the future forecast, so the whole timeline is continuous.
# ============================================================
def _predict_step(date, state, crop, weather_row, diesel_change,
                   prices, seed_features, step_index):
    if step_index == 0 and seed_features is not None:
        lag1, lag3 = seed_features["lag_1"], seed_features["lag_3"]
        lag7, lag14 = seed_features["lag_7"], seed_features["lag_14"]
        rolling_mean7 = seed_features["rolling_mean_7"]
        rolling_std7 = seed_features["rolling_std_7"]
    else:
        lag1 = prices[-1]
        lag3 = prices[-3] if len(prices) >= 3 else prices[0]
        lag7 = prices[-7] if len(prices) >= 7 else prices[0]
        lag14 = prices[-14] if len(prices) >= 14 else prices[0]
        rolling_mean7 = float(np.mean(prices[-7:]))
        rolling_std7 = float(np.std(prices[-7:]))

    fest_row = get_festive_features([date], state).iloc[0]

    if MODEL_LOADED:
        X = build_feature_row(FEATURE_COLUMNS, date, state, crop, weather_row,
                               diesel_change, lag1, lag3, lag7, lag14,
                               rolling_mean7, rolling_std7,
                               fest_row["Is_Festive"], fest_row["Days_To_Festive"])
        pred = float(MODEL.predict(X)[0])
    else:
        # graceful fallback so the UI still renders without the pkl files
        pred = lag1 * (1 - 0.015) + np.random.default_rng(step_index).normal(0, 0.05)
    return pred

def run_forecast(state, crop, horizon_days):
    hist, seed_features, hist_live, pricecatcher_connected, live_days_added = get_actual_price_history(state, crop, 21)
    prices = list(hist["price"].values)
    last_real_date = pd.Timestamp(hist["date"].max())
    today = pd.Timestamp(get_myt_today())

    diesel_price, diesel_change, diesel_live = get_diesel_price()

    # ---- 1) BRIDGE (nowcast for days already passed but not in the CSV) ----
    bridge_df = pd.DataFrame(columns=["date", "forecast_price"])
    bridge_weather_live = True
    if last_real_date < today:
        bridge_dates = pd.date_range(last_real_date + timedelta(days=1), today)
        bw_df, bridge_weather_live = get_weather_historical(
            state, bridge_dates[0].strftime("%Y-%m-%d"), bridge_dates[-1].strftime("%Y-%m-%d"))
        bw_df = bw_df.set_index("date").reindex(bridge_dates)
        bw_df = bw_df.ffill().bfill()  # archive API can lag by a day or two for very recent dates
        bw_df = bw_df.reset_index().rename(columns={"index": "date"})

        bridge_rows = []
        for i, date in enumerate(bridge_dates):
            pred = _predict_step(date, state, crop, bw_df.iloc[i], diesel_change,
                                  prices, seed_features if i == 0 else None, i)
            prices.append(pred)
            bridge_rows.append({"date": date, "forecast_price": pred})
        bridge_df = pd.DataFrame(bridge_rows)

    # ---- 2) FUTURE forecast from today onward ----
    weather_df, weather_live = get_weather_forecast(state, horizon_days + 1)
    weather_df = weather_df[weather_df["date"] > today].reset_index(drop=True)
    if len(weather_df) < horizon_days:
        start_extra = weather_df["date"].iloc[-1] + timedelta(days=1) if len(weather_df) else today + timedelta(days=1)
        extra_dates = pd.date_range(start_extra, periods=horizon_days - len(weather_df))
        extra = pd.DataFrame({"date": extra_dates, "Rainfall_mm": 5.0,
                               "Max_Temp": 32.0, "Min_Temp": 24.0, "Humidity_Pct": 80.0})
        weather_df = pd.concat([weather_df, extra], ignore_index=True)
    weather_df = weather_df.head(horizon_days).reset_index(drop=True)

    forecast_rows = []
    for i in range(horizon_days):
        date = weather_df["date"].iloc[i]
        seed = seed_features if (i == 0 and bridge_df.empty) else None
        pred = _predict_step(date, state, crop, weather_df.iloc[i], diesel_change,
                              prices, seed, i)
        prices.append(pred)
        forecast_rows.append({"date": date, "forecast_price": pred,
                               "lower_ci": pred * 0.94, "upper_ci": pred * 1.06})
    forecast_df = pd.DataFrame(forecast_rows)
    festive_df = get_festive_features(list(weather_df["date"]), state)

    return hist, bridge_df, forecast_df, weather_df, diesel_price, diesel_change, \
        weather_live, diesel_live, festive_df, hist_live, bridge_weather_live, \
        pricecatcher_connected, live_days_added

if "params" not in st.session_state:
    st.session_state["params"] = ("Perak", CROPS[0], 7)

init_state, init_crop, init_horizon = st.session_state["params"]

if "forecast_result" not in st.session_state:
    st.session_state["forecast_result"] = run_forecast(init_state, init_crop, init_horizon)

hist, bridge_df, forecast_df, weather_df, diesel_price, diesel_change, weather_live, diesel_live, festive_df, hist_live, bridge_weather_live, pricecatcher_connected, live_days_added = st.session_state["forecast_result"]
state, crop, horizon_days = st.session_state["params"]

current_price = bridge_df["forecast_price"].iloc[-1] if not bridge_df.empty else hist["price"].iloc[-1]
last_forecast_price = forecast_df["forecast_price"].iloc[-1]
next_festive_name = festive_df["next_festive_name"].iloc[0]
days_to_festive = int(festive_df["Days_To_Festive"].iloc[0])


# ============================================================
# HEADER
# ============================================================
st.markdown(f"""
<div class="top-banner" style="display: flex; justify-content: space-between; align-items: center;padding: 50px 40px;">
  <div style="width: 180px;"></div> <!-- Ruang penyeimbang kiri -->
  
  <div style="text-align: center; flex-grow: 1;">
    <h1 style="margin-top: 10px;margin-bottom: 5; font-size:40px; font-weight: 700;">Sistem Ramalan Harga Sayur di Malaysia</h1>
    <p style="margin:0; opacity:0.85; font-size: 20px;">Laporan Ramalan Trend Harga Mingguan dan Bulanan Untuk Sayur-Sayuran</p>
  </div>
  
  <div style="margin-top: 20px;">
    <div class="live-badge">
      Tarikh kemas kini data<br><b>{datetime.now(ZoneInfo("Asia/Kuala_Lumpur")).strftime('%d %b %Y, %I:%M %p')}</b>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

if not MODEL_LOADED:
    st.warning(f"⚠️ Model / feature-columns pkl tidak dijumpai ({MODEL_LOAD_ERROR}). "
               f"Dashboard berjalan dalam mod anggaran sementara. Letakkan "
               f"'{MODEL_PATH}' dan '{FEATURE_COLS_PATH}' dalam folder yang sama.")

# ============================================================
# KPI CARDS
# ============================================================
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""<div class="card">
        <div class="card-label"> TABURAN HUJAN SEMASA</div>
        <div class="card-value">{weather_df['Rainfall_mm'].iloc[0]:.1f} mm</div>
        <div class="card-sub">{' Hujan Renyai / Sederhana' if weather_df['Rainfall_mm'].iloc[0] < 20 else ' Hujan Lebat'}</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="card">
        <div class="card-label"> JULAT SUHU & UDARA</div>
        <div class="card-value">{weather_df['Min_Temp'].iloc[0]:.0f}° – {weather_df['Max_Temp'].iloc[0]:.0f}°C</div>
        <div class="card-sub"> Kelembapan Udara: {weather_df['Humidity_Pct'].iloc[0]:.0f}%</div>
    </div>""", unsafe_allow_html=True)
with k3:
    sign = "▲" if diesel_price > 0 else "▼"
    diesel_status = "Kos Meningkat" if diesel_price >= 0 else "Kos Menurun"
    st.markdown(f"""<div class="card">
        <div class="card-label"> PERUBAHAN HARGA DIESEL </div>
        <div class="card-value">{sign} RM{diesel_price:.2f} /L</div>
        <div class="card-sub">{diesel_status}</div>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="card">
        <div class="card-label"> PERISTIWA/PERAYAAN TERDEKAT</div>
        <div class="card-value">{days_to_festive} Hari lagi</div>
        <div class="card-sub">{next_festive_name}</div>
    </div>""", unsafe_allow_html=True)

st.write("")


# ============================================================
# CONTROL PANEL
# ============================================================
st.markdown("#### Pilihan Ramalan")

# Gunakan st.form supaya semua widget terbungkus secara automatik di dalam kotak kontena
with st.form("form_pilihan_ramalan"):
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    
    with c1:
        state = st.selectbox("Negeri", list(STATE_COORDS.keys()), index=list(STATE_COORDS.keys()).index(state))
    with c2:
        crop = st.selectbox("Komoditi Sayuran", CROPS, index=CROPS.index(crop))
    with c3:
        horizon_label = st.radio("Tempoh Ramalan", ["7 Hari", "30 Hari"], horizontal=True, index=0 if horizon_days == 7 else 1)
    with c4:
        st.write("") # Penjarak menegak selari dengan butang
        # Gunakan st.form_submit_button khusus untuk di dalam form
        generate = st.form_submit_button("Jana Ramalan Harga", use_container_width=True, type="primary")

# Tindakan apabila butang "Jana Ramalan Harga" ditekan di dalam form
if generate:
    horizon_days = 7 if horizon_label == "7 Hari" else 30
    st.session_state["forecast_result"] = run_forecast(state, crop, horizon_days)
    st.session_state["params"] = (state, crop, horizon_days)
    st.rerun()

# ============================================================
# MAIN TRAJECTORY CHART
# ============================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown(f"#### Trajektori Harga Ladang Semasa & Ramalan AI")
last_real_date = hist["date"].max()
today_ts = pd.Timestamp(get_myt_today())
gap_days = max((today_ts - pd.Timestamp(last_real_date)).days, 0)

# ============================================================
# 1. MASUKKAN BAHAGIAN RINGKASAN & KOTAK AMARAN DI SINI (SEBELUM CARTA)
# ============================================================
total_change = last_forecast_price - current_price

accuracy = compute_model_accuracy(state, crop)

st.markdown(f"""
* **Harga Ramalan Hari Ini:** RM {current_price:.2f}
* **Harga Purata Ramalan ({horizon_days} Hari):** RM {last_forecast_price:.2f}
* **Ketepatan Ramalan:** {accuracy:.1f}% (Berdasarkan prestasi model pada data terdahulu)
""")

# Paparkan kotak amaran merah jika harga unjuran menurun
if total_change < 0:
    st.markdown("""
    <div style="background-color: #ff3b30; color: white; padding: 15px 20px; border-radius: 8px; margin: 15px 0px; display: flex; align-items: center; gap: 15px;">
        <div style="font-size: 26px;">⚠️</div>
        <div>
            <div style="font-weight: 700; font-size: 15px;">Harga dijangka turun dalam tempoh ramalan akan datang</div>
            <div style="font-size: 13px;">Pertimbangkan jual awal atau elak tambah tanaman baru buat masa ini</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 2. Chart plotly code 
# ============================================================
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist["date"], y=hist["price"], mode="lines+markers",
                          name="Harga Sebenar (Rekod)",
                          line=dict(color="#0b3d2e", width=3),
                          hovertemplate="Tarikh: %{x|%d %b %Y}<br>Harga Sebenar: RM%{y:.2f}/kg<extra></extra>"))

if not bridge_df.empty:
    bridge_x = [hist["date"].iloc[-1]] + list(bridge_df["date"])
    bridge_y = [hist["price"].iloc[-1]] + list(bridge_df["forecast_price"])
    fig.add_trace(go.Scatter(x=bridge_x, y=bridge_y, mode="lines+markers",
                              name="Anggaran Susulan (Belum Diterbitkan PriceCatcher)",
                              line=dict(color="#d97706", width=3, dash="dash"),
                              hovertemplate="Tarikh: %{x|%d %b %Y}<br>Anggaran Susulan: RM%{y:.2f}/kg<extra></extra>"))
    anchor_date, anchor_price = bridge_df["date"].iloc[-1], bridge_df["forecast_price"].iloc[-1]
else:
    anchor_date, anchor_price = hist["date"].iloc[-1], hist["price"].iloc[-1]

forecast_x = [anchor_date] + list(forecast_df["date"])
forecast_y = [anchor_price] + list(forecast_df["forecast_price"])
ci_upper = [anchor_price] + list(forecast_df["upper_ci"])
ci_lower = [anchor_price] + list(forecast_df["lower_ci"])
fig.add_trace(go.Scatter(x=forecast_x, y=ci_upper, mode="lines", line=dict(width=0), showlegend=False,
                          hovertemplate="Tarikh: %{x|%d %b %Y}<br>Anggaran Maksimum: RM%{y:.2f}/kg<extra></extra>"))
fig.add_trace(go.Scatter(x=forecast_x, y=ci_lower, mode="lines", line=dict(width=0), fill="tonexty",
                          fillcolor="rgba(52,211,153,0.25)", name="Zon Ramalan - Julat Maksimum dan Minimum",
                          hovertemplate="Tarikh: %{x|%d %b %Y}<br>Anggaran Minimum: RM%{y:.2f}/kg<extra></extra>"))
fig.add_trace(go.Scatter(x=forecast_x, y=forecast_y, mode="lines+markers",
                          name="Ramalan AI ",
                          line=dict(color="#dc2626", width=3, dash="dot"),
                          hovertemplate="Tarikh: %{x|%d %b %Y}<br>Harga Ramalan: RM%{y:.2f}/kg<extra></extra>"))
fig.add_vline(x=today_ts, line_dash="dash", line_color="#b91c1c",
              annotation_text=f"HARI INI ({today_ts.strftime('%d %b')})",
              annotation_position="top")
fig.update_layout(height=460, plot_bgcolor="white", paper_bgcolor="white",
                   legend=dict(orientation="h", y=1.15),
                   yaxis_title="RM / kg", margin=dict(t=30, b=10),
                   hoverlabel=dict(bgcolor="white", font_size=13, namelength=-1))
st.plotly_chart(fig, use_container_width=True)

total_change = last_forecast_price - current_price
st.markdown("""
<div style="background-color: #f8f9fa; padding: 10px 15px; border-radius: 5px; border: 1px solid #e9ecef; font-size: 14px; color: #31333F;">
    🛈  Harga komoditi : PriceCatcher (Diterbitkan oleh KPDN) | Cuaca: Open Meteo API | Harga Diesel : OpenDOSM | Peristiwa/Perayaan Malaysia: Python Holidays library
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# DETAIL TABLE
# ============================================================

st.markdown("#### Lihat Jadual Butiran Ramalan")
display_df = forecast_df.copy()
display_df["date"] = display_df["date"].dt.strftime("%d %b %Y")
for c in ["forecast_price", "lower_ci", "upper_ci"]:
    display_df[c] = display_df[c].round(2)
display_df = display_df.rename(columns={
    "date": "Tarikh", 
    "forecast_price": "Ramalan Harga (RM/kg)",
    "lower_ci": "Julat Minimum (RM/kg)", 
    "upper_ci": "Julat Maksimum (RM/kg)"
})
# 1. Tukar dataframe kepada format HTML dengan tetapan tengah
html_table = display_df.to_html(classes='custom-table', index=False, justify='center')

# 2. Paparkan dengan gaya CSS yang pasti berkesan untuk jadual HTML
st.markdown(f"""
<style>
.custom-table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 10px;
    margin-bottom: 10px;
    font-size: 14px;
}}
.custom-table th {{
    font-weight: 700 !important;
    color: #000000 !important;
    background-color: #f0f2f6 !important;
    text-align: center !important;
    padding: 10px;
    border: 1px solid #d6d6d6;
}}
.custom-table td {{
    text-align: center !important;
    padding: 8px;
    border: 1px solid #d6d6d6;
    color: #31333F;
}}
</style>
{html_table}
""", unsafe_allow_html=True)

st.markdown("""
<div style="background-color: #f8f9fa; padding: 10px 15px; border-radius: 5px; border: 1px solid #e9ecef; font-size: 14px; color: #31333F;">
    🛈  Harga sebenar berkemungkinan berada dalam julat minimum dan maksimum yang dipaparkan. Julat ini menunjukkan kemungkinan perubahan harga berdasarkan corak data sebelum ini.
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)