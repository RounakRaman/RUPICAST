# =============================================================================
# USD/INR MACRO INTELLIGENCE DASHBOARD
# Streamlit App with Live Data, ARIMAX Forecasting, AI Scenario Analysis,
# Seasonality Decomposition, and Confidence Interval Visualizations
# =============================================================================
# REQUIREMENTS (pip install these):
#   streamlit yfinance pandas numpy matplotlib seaborn statsmodels
#   scikit-learn plotly anthropic pandas_datareader requests beautifulsoup4
# =============================================================================
# RUN: streamlit run usd_inr_dashboard.py
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import requests
from datetime import datetime, timedelta
from pandas.tseries.offsets import MonthBegin
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
import anthropic
import json
import io
import time

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="USD/INR Macro Intelligence",
    page_icon="₹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS (dark financial terminal aesthetic) ────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0e17;
    color: #e2e8f0;
  }
  .stApp { background: #0a0e17; }

  /* Header */
  .main-header {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #f7c948 0%, #f97316 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
    letter-spacing: -1px;
  }
  .sub-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #64748b;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 2px;
  }

  /* Metric cards */
  .metric-card {
    background: #111827;
    border: 1px solid #1e2d45;
    border-left: 3px solid #f7c948;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
  }
  .metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #64748b;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 4px;
  }
  .metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #f7c948;
  }
  .metric-delta-up   { color: #22c55e; font-size: 0.8rem; }
  .metric-delta-down { color: #ef4444; font-size: 0.8rem; }

  /* Section headers */
  .section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #f7c948;
    border-bottom: 1px solid #1e2d45;
    padding-bottom: 8px;
    margin: 24px 0 16px;
  }

  /* AI response box */
  .ai-response {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 20px 24px;
    margin-top: 16px;
    font-size: 0.92rem;
    line-height: 1.7;
    color: #cbd5e1;
  }

  /* Scenario pills */
  .scenario-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    font-weight: 700;
    margin-right: 8px;
  }
  .pill-bull  { background: #052e16; color: #4ade80; border: 1px solid #166534; }
  .pill-base  { background: #1e1b4b; color: #818cf8; border: 1px solid #3730a3; }
  .pill-bear  { background: #3b0764; color: #e879f9; border: 1px solid #7e22ce; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #1e2d45;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    background: #0d1117;
    border-bottom: 1px solid #1e2d45;
    gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 1px;
    color: #64748b;
    background: transparent;
    border: none;
    padding: 8px 16px;
  }
  .stTabs [aria-selected="true"] {
    color: #f7c948 !important;
    border-bottom: 2px solid #f7c948 !important;
    background: transparent !important;
  }

  /* Plotly chart background match */
  .js-plotly-plot { border-radius: 8px; }

  /* Input styling */
  .stTextArea textarea {
    background: #0d1117;
    border: 1px solid #1e2d45;
    color: #e2e8f0;
    font-family: 'DM Sans', sans-serif;
  }
  .stSlider > div { color: #e2e8f0; }
  div[data-testid="stMetricValue"] { font-family: 'Space Mono', monospace; }

  /* Live badge */
  .live-badge {
    display: inline-block;
    background: #052e16;
    color: #4ade80;
    border: 1px solid #166534;
    border-radius: 20px;
    padding: 2px 10px;
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 1px;
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.5; }
  }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: PLOTLY THEME
# ═══════════════════════════════════════════════════════════════════════════════
PLOTLY_LAYOUT = dict(
    plot_bgcolor="#0d1117",
    paper_bgcolor="#0d1117",
    font=dict(family="Space Mono, monospace", color="#94a3b8", size=11),
    xaxis=dict(gridcolor="#1e2d45", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#1e2d45", showgrid=True, zeroline=False),
    legend=dict(bgcolor="#0d1117", bordercolor="#1e2d45", borderwidth=1),
    margin=dict(l=20, r=20, t=40, b=20),
)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)   # refresh every hour
def load_live_usdinr(period_years: int = 10) -> pd.DataFrame:
    """Fetch daily USD/INR from Yahoo Finance, resample to monthly."""
    end   = datetime.today()
    start = end - timedelta(days=365 * period_years)
    ticker = yf.Ticker("USDINR=X")
    df = ticker.history(start=start, end=end, interval="1d")
    if df.empty:
        st.error("Could not fetch live USD/INR data. Check internet connection.")
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[["Close"]].rename(columns={"Close": "USD_INR"})
    monthly = df.resample("MS").mean()   # Month Start
    return monthly


@st.cache_data(ttl=86400)
def load_macro_fred(start="2014-01-01", api_key: str = ""):
    """
    Fetch FRED macro series via direct REST API (requires free FRED API key).
    Falls back gracefully series-by-series.
    Returns (DataFrame | None, pct_fetched 0.0-1.0).
    """
    if not api_key:
        return None, 0.0

    SERIES = {
        "CPI_USA":             "MEDCPIM158SFRBCLE",
        "Crude_Oil":           "DCOILWTICO",
        "Trade_Balance_India": "XTEXVA01INM667S",
        "US_Rate_EFFR":        "FEDFUNDS",
    }

    end_date   = datetime.today().strftime("%Y-%m-%d")
    frames = {}

    for col, series_id in SERIES.items():
        try:
            url = (
                f"https://api.stlouisfed.org/fred/series/observations"
                f"?series_id={series_id}"
                f"&observation_start={start}"
                f"&observation_end={end_date}"
                f"&api_key={api_key}"
                f"&file_type=json"
            )
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            obs  = resp.json().get("observations", [])
            if not obs:
                continue
            df_s = pd.DataFrame(obs)[["date", "value"]]
            df_s["value"] = pd.to_numeric(df_s["value"], errors="coerce")
            df_s["date"]  = pd.to_datetime(df_s["date"])
            df_s = df_s.dropna().set_index("date")["value"].rename(col)
            frames[col] = df_s
        except Exception:
            pass

    if not frames:
        return None, 0.0

    fred = pd.concat(frames.values(), axis=1)
    fred.index = fred.index - MonthBegin(1)
    for c in ["CPI_USA", "Crude_Oil"]:
        if c in fred.columns:
            fred[c] = fred[c].interpolate(method="linear", limit_direction="both")
    fred = fred.resample("MS").mean()

    pct = len(frames) / len(SERIES)
    return fred, pct


def build_synthetic_macro(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate plausible synthetic macro data when live sources are unavailable."""
    n = len(idx)
    np.random.seed(42)
    df = pd.DataFrame(index=idx)
    df["CPI_USA"]             = 2.5 + np.cumsum(np.random.normal(0.05, 0.1, n))
    df["Crude_Oil"]           = 70  + np.cumsum(np.random.normal(0,    1.5, n))
    df["Trade_Balance_India"] = -15 + np.random.normal(0, 2, n)
    df["US_Rate_EFFR"]        = np.clip(
        5.33 - np.linspace(0, 2, n) + np.random.normal(0, 0.1, n), 0, 8
    )
    df["RBI_Repo_Rate"]       = np.clip(
        6.5  - np.linspace(0, 1, n) + np.random.normal(0, 0.1, n), 4, 9
    )
    df["Total_Reserves_USD"]  = 600 + np.cumsum(np.random.normal(1, 5, n))
    df["Ind_CPI"]             = 5   + np.random.normal(0, 0.5, n)
    return df


@st.cache_data(ttl=86400)
def load_rbi_repo_rate() -> tuple:
    """
    Fetch RBI Repo Rate history from RBI's DBIE portal.
    Tries multiple public RBI endpoints and falls back gracefully.
    Returns (Series | None, source_label).
    """
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/html, */*",
    }

    # ── Attempt 1: RBI DBIE REST API (repo rate series ID: II-B-1) ──
    try:
        url = (
            "https://api.rbi.org.in/api/v3/findbyfacets?"
            "facets=B02&startDate=2010-01-01"
            f"&endDate={datetime.today().strftime('%Y-%m-%d')}"
            "&frequency=M&lang=EN"
        )
        resp = requests.get(url, headers=HEADERS, timeout=12)
        data = resp.json()
        records = data.get("data", data.get("Data", []))
        if records:
            df = pd.DataFrame(records)
            # column names vary — try to find date + value cols
            date_col  = next((c for c in df.columns if "date" in c.lower()), None)
            val_col   = next((c for c in df.columns if any(
                k in c.lower() for k in ["repo","rate","value","val"])), None)
            if date_col and val_col:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df[val_col]  = pd.to_numeric(df[val_col], errors="coerce")
                s = df.dropna(subset=[date_col, val_col]).set_index(date_col)[val_col]
                s.index = s.index - MonthBegin(1)
                s = s.resample("MS").last().rename("RBI_Repo_Rate")
                if len(s) > 12:
                    return s, "RBI DBIE API"
    except Exception:
        pass

    # ── Attempt 2: RBI DBIE CSV download (key monetary rates table) ──
    try:
        csv_url = (
            "https://rbidbie.rbi.org.in/scripts/BS_NSDPDisplay.aspx"
            "?param=B&Series=B02&DateRange=2010-2025&Language=EN&output=csv"
        )
        resp = requests.get(csv_url, headers=HEADERS, timeout=15)
        df   = pd.read_csv(io.StringIO(resp.text), skiprows=2)
        df.columns = [c.strip() for c in df.columns]
        date_col = df.columns[0]
        rate_col = next((c for c in df.columns if "repo" in c.lower()), df.columns[1])
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
        df[rate_col] = pd.to_numeric(
            df[rate_col].astype(str).str.replace("%","").str.strip(), errors="coerce"
        )
        s = df.dropna(subset=[date_col, rate_col]).set_index(date_col)[rate_col]
        s.index = s.index - MonthBegin(1)
        s = s.resample("MS").last().rename("RBI_Repo_Rate")
        if len(s) > 12:
            return s, "RBI DBIE CSV"
    except Exception:
        pass

    # ── Attempt 3: Wikipedia / known public table ──
    try:
        wiki_url = "https://en.wikipedia.org/wiki/Repo_rate_in_India"
        resp  = requests.get(wiki_url, headers=HEADERS, timeout=12)
        dfs   = pd.read_html(io.StringIO(resp.text))
        for df in dfs:
            df.columns = [str(c).lower().strip() for c in df.columns]
            if any("repo" in c or "rate" in c for c in df.columns):
                date_col = next((c for c in df.columns if "date" in c or "year" in c), None)
                rate_col = next((c for c in df.columns if "repo" in c or "rate" in c), None)
                if date_col and rate_col:
                    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
                    df[rate_col] = pd.to_numeric(
                        df[rate_col].astype(str).str.replace("%",""), errors="coerce"
                    )
                    s = df.dropna(subset=[date_col, rate_col]).set_index(date_col)[rate_col]
                    s.index = s.index - MonthBegin(1)
                    s = s.resample("MS").last().ffill().rename("RBI_Repo_Rate")
                    if len(s) > 12:
                        return s, "Wikipedia (public)"
    except Exception:
        pass

    return None, "Synthetic"


@st.cache_data(ttl=86400)
def load_rbi_fx_reserves() -> tuple:
    """
    Fetch India FX Reserves from RBI DBIE or FRED (as fallback).
    Returns (Series | None, source_label).
    """
    HEADERS = {"User-Agent": "Mozilla/5.0"}

    # ── Try FRED first (RESIRUSD = India total reserves) ──
    try:
        fred_key = st.session_state.get("fred_api_key", "")
        if fred_key:
            url = (
                f"https://api.stlouisfed.org/fred/series/observations"
                f"?series_id=RESIRUSD&observation_start=2010-01-01"
                f"&api_key={fred_key}&file_type=json"
            )
            resp = requests.get(url, timeout=12)
            obs  = resp.json().get("observations", [])
            if obs:
                df = pd.DataFrame(obs)[["date","value"]]
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                df["date"]  = pd.to_datetime(df["date"])
                s = df.dropna().set_index("date")["value"] / 1e9  # convert to USD bn
                s.index = s.index - MonthBegin(1)
                s = s.resample("MS").last().rename("Total_Reserves_USD")
                if len(s) > 12:
                    return s, "FRED (RESIRUSD)"
    except Exception:
        pass

    return None, "Synthetic"


@st.cache_data(ttl=86400)
def load_india_cpi_rbi(fred_key: str = "") -> tuple:
    """
    Fetch India CPI from FRED series INDCPIALLMINMEI.
    Returns (Series | None, source_label).
    """
    if not fred_key:
        return None, "Synthetic"
    try:
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id=INDCPIALLMINMEI&observation_start=2010-01-01"
            f"&api_key={fred_key}&file_type=json"
        )
        resp = requests.get(url, timeout=12)
        obs  = resp.json().get("observations", [])
        if obs:
            df = pd.DataFrame(obs)[["date","value"]]
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df["date"]  = pd.to_datetime(df["date"])
            s = df.dropna().set_index("date")["value"]
            # convert index level to YoY %
            s = s.pct_change(12) * 100
            s.index = s.index - MonthBegin(1)
            s = s.resample("MS").last().rename("Ind_CPI")
            if len(s) > 12:
                return s, "FRED (INDCPIALLMINMEI)"
    except Exception:
        pass
    return None, "Synthetic"


@st.cache_data(ttl=3600)
def get_live_spot_rate() -> dict:
    """Get current USD/INR spot from Yahoo Finance."""
    try:
        t  = yf.Ticker("USDINR=X")
        h  = t.history(period="2d")
        if not h.empty:
            latest = float(h["Close"].iloc[-1])
            prev   = float(h["Close"].iloc[-2]) if len(h) > 1 else latest
            return {"rate": latest, "prev": prev, "change": latest - prev,
                    "pct": (latest - prev) / prev * 100, "ok": True}
    except Exception:
        pass
    return {"rate": None, "ok": False}


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def run_adf(series: pd.Series) -> dict:
    s = series.dropna()
    if len(s) < 10 or s.max() == s.min():
        return {"stat": None, "pval": None, "stationary": None}
    try:
        r = adfuller(s)
        return {"stat": round(r[0], 4), "pval": round(r[1], 4),
                "stationary": r[1] < 0.05}
    except Exception:
        return {"stat": None, "pval": None, "stationary": None}


def fit_arimax(y_diff, exog_scaled, order=(1, 1, 0)) -> SARIMAX:
    m = SARIMAX(y_diff, exog=exog_scaled, order=order,
                enforce_stationarity=False, enforce_invertibility=False)
    return m.fit(disp=False)


def forecast_levels(result, exog_future_scaled, last_level, steps=6) -> pd.DataFrame:
    fc   = result.get_forecast(steps=steps, exog=exog_future_scaled)
    mean = fc.predicted_mean
    ci   = fc.conf_int()
    lvl  = last_level + mean.cumsum()
    lo   = last_level + (mean + (ci.iloc[:, 0] - mean)).cumsum()
    hi   = last_level + (mean + (ci.iloc[:, 1] - mean)).cumsum()
    return pd.DataFrame({"Forecast": lvl, "Lower": lo, "Upper": hi}, index=mean.index)


# ═══════════════════════════════════════════════════════════════════════════════
# AI HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def ai_scenario_analysis(news_text: str, macro_context: dict,
                          current_rate: float, api_key: str) -> dict:
    """
    Call Claude to extract macro signals from news and return
    Bull / Base / Bear scenario forecasts with reasoning.
    Returns a dict with keys: bull, base, bear, summary, signals.
    """
    client = anthropic.Anthropic(api_key=api_key)

    system = """You are a senior FX macro strategist specialising in USD/INR.
You receive: (a) news/commentary, (b) live macro indicators, (c) current spot rate.
Return ONLY valid JSON with exactly this schema (no markdown, no extra text):
{
  "signals": ["signal1", "signal2", ...],
  "summary": "2-3 sentence macro narrative",
  "bull": {
    "label": "Rupee Appreciation",
    "direction": "INR strengthens",
    "target_3m": <float>,
    "target_6m": <float>,
    "probability_pct": <int>,
    "reasoning": "..."
  },
  "base": {
    "label": "Range-Bound",
    "direction": "Sideways",
    "target_3m": <float>,
    "target_6m": <float>,
    "probability_pct": <int>,
    "reasoning": "..."
  },
  "bear": {
    "label": "Rupee Depreciation",
    "direction": "INR weakens",
    "target_3m": <float>,
    "target_6m": <float>,
    "probability_pct": <int>,
    "reasoning": "..."
  }
}
Probabilities must sum to 100. Be realistic. Current date: """ + datetime.today().strftime("%B %Y")

    user_msg = f"""NEWS / COMMENTARY:
{news_text}

MACRO CONTEXT:
{json.dumps(macro_context, indent=2)}

CURRENT USD/INR SPOT: {current_rate:.4f}

Provide your scenario analysis."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system=system,
        messages=[{"role": "user", "content": user_msg}]
    )
    raw = message.content[0].text.strip()
    # strip json fences if present
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def ai_quantify_scenario(scenario: str, macro_deltas: dict,
                          result, exog_scaled, last_level: float,
                          steps: int = 6) -> pd.DataFrame:
    """
    Translate AI scenario (bull/base/bear) into quantified macro shocks
    and re-run the ARIMAX model with perturbed exogenous variables.
    """
    shocks = {
        "bull": {"US_Rate_EFFR": -0.25, "Crude_Oil": -5,
                 "Total_Reserves_USD": 10, "CPI_USA": -0.1},
        "base": {"US_Rate_EFFR":  0.00, "Crude_Oil":  0,
                 "Total_Reserves_USD":  0, "CPI_USA":  0.0},
        "bear": {"US_Rate_EFFR": +0.25, "Crude_Oil": +8,
                 "Total_Reserves_USD": -8, "CPI_USA": +0.2},
    }
    delta = shocks.get(scenario, shocks["base"])
    future_exog = exog_scaled.iloc[-steps:].copy()
    for col, shock in delta.items():
        if col in future_exog.columns:
            future_exog[col] = future_exog[col] + shock

    fc  = result.get_forecast(steps=steps, exog=future_exog)
    mean = fc.predicted_mean
    ci   = fc.conf_int()
    lvl  = last_level + mean.cumsum()
    lo   = last_level + (mean + (ci.iloc[:, 0] - mean)).cumsum()
    hi   = last_level + (mean + (ci.iloc[:, 1] - mean)).cumsum()
    return pd.DataFrame({"Forecast": lvl, "Lower": lo, "Upper": hi},
                        index=mean.index)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_usdinr(hist: pd.Series, forecast_df: pd.DataFrame = None,
                title="USD/INR Historical & Forecast") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist.values,
        name="Actual USD/INR", line=dict(color="#f7c948", width=2)
    ))
    if forecast_df is not None and not forecast_df.empty:
        fig.add_trace(go.Scatter(
            x=forecast_df.index, y=forecast_df["Forecast"],
            name="Forecast", line=dict(color="#f97316", width=2, dash="dash")
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df.index.to_series(),
                         forecast_df.index.to_series()[::-1]]),
            y=pd.concat([forecast_df["Upper"], forecast_df["Lower"][::-1]]),
            fill="toself", fillcolor="rgba(249,115,22,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="95% CI",
            showlegend=True
        ))
        # vertical dotted line at forecast start
        fig.add_vline(x=str(forecast_df.index[0]),
                      line=dict(color="#475569", dash="dot", width=1))
    fig.update_layout(title=title, **PLOTLY_LAYOUT)
    return fig


def plot_scenarios(hist: pd.Series, scenarios: dict) -> go.Figure:
    """scenarios = {'bull': df, 'base': df, 'bear': df}"""
    COLORS = {"bull": "#4ade80", "base": "#818cf8", "bear": "#e879f9"}
    LABELS = {"bull": "🟢 Bull — INR Strengthens",
              "base": "🟣 Base — Range-Bound",
              "bear": "🔴 Bear — INR Weakens"}
    fig = go.Figure()
    # Historical
    fig.add_trace(go.Scatter(
        x=hist.index[-36:], y=hist.values[-36:],
        name="Historical", line=dict(color="#f7c948", width=2)
    ))
    for key, df in scenarios.items():
        c = COLORS[key]
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Forecast"],
            name=LABELS[key], line=dict(color=c, width=2, dash="dot")
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([df.index.to_series(), df.index.to_series()[::-1]]),
            y=pd.concat([df["Upper"], df["Lower"][::-1]]),
            fill="toself",
            fillcolor=f"rgba{tuple(list(int(c.lstrip('#')[i:i+2], 16) for i in (0,2,4)) + [0.08])}",
            line=dict(color="rgba(0,0,0,0)"),
            name=f"{key} CI", showlegend=False
        ))
    fig.update_layout(title="USD/INR Scenario Fan Chart", **PLOTLY_LAYOUT)
    return fig


def plot_seasonality(monthly: pd.Series) -> go.Figure:
    df = monthly.to_frame("USD_INR")
    df["Month"] = df.index.month
    df["Year"]  = df.index.year
    monthly_avg = df.groupby("Month")["USD_INR"].mean()
    grand_mean  = df["USD_INR"].mean()
    seasonal    = monthly_avg - grand_mean

    fig = make_subplots(rows=2, cols=2,
        subplot_titles=["Monthly Seasonal Pattern", "YoY Returns by Month",
                        "Distribution of Monthly Returns", "Seasonal Decompose – Trend"])
    # 1. Bar chart seasonality
    colors = ["#4ade80" if v < 0 else "#ef4444" for v in seasonal.values]
    fig.add_trace(go.Bar(
        x=["Jan","Feb","Mar","Apr","May","Jun",
           "Jul","Aug","Sep","Oct","Nov","Dec"],
        y=seasonal.values, marker_color=colors, name="Seasonal Deviation"
    ), row=1, col=1)

    # 2. Year-on-year monthly returns heatmap
    df2 = df.copy()
    df2["Return"] = df2["USD_INR"].pct_change() * 100
    pivot = df2.pivot_table(values="Return", index="Year", columns="Month")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]
    fig.add_trace(go.Heatmap(
        z=pivot.values, x=list(pivot.columns), y=list(pivot.index),
        colorscale="RdYlGn_r", name="MoM %",
        colorbar=dict(x=0.48, thickness=10)
    ), row=1, col=2)

    # 3. Distribution of monthly changes
    df2.dropna(subset=["Return"], inplace=True)
    fig.add_trace(go.Histogram(
        x=df2["Return"], nbinsx=30,
        marker_color="#f7c948", opacity=0.8, name="Return dist."
    ), row=2, col=1)

    # 4. Trend (simple decompose)
    try:
        dec = seasonal_decompose(monthly.dropna(), model="additive", period=12)
        fig.add_trace(go.Scatter(
            x=dec.trend.index, y=dec.trend.values,
            line=dict(color="#f97316", width=2), name="Trend"
        ), row=2, col=2)
    except Exception:
        pass

    fig.update_layout(height=700, showlegend=False,
                      title="Seasonality & Return Analysis", **PLOTLY_LAYOUT)
    return fig


def plot_correlation(df_scaled: pd.DataFrame) -> go.Figure:
    corr = df_scaled.corr()
    fig  = go.Figure(go.Heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        colorscale="RdBu",
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        colorbar=dict(thickness=14)
    ))
    fig.update_layout(title="Macro Correlation Matrix", height=520, **PLOTLY_LAYOUT)
    return fig


def plot_residuals(resid: pd.Series) -> go.Figure:
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=["Residuals Over Time", "Residual Histogram"])
    fig.add_trace(go.Scatter(
        x=resid.index, y=resid.values,
        line=dict(color="#94a3b8", width=1), name="Residuals"
    ), row=1, col=1)
    fig.add_trace(go.Histogram(
        x=resid.values, marker_color="#f7c948", opacity=0.8
    ), row=1, col=2)
    fig.update_layout(height=340, showlegend=False,
                      title="ARIMAX Model Residuals", **PLOTLY_LAYOUT)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="section-title">⚙ CONFIGURATION</div>',
                unsafe_allow_html=True)

    anthropic_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Required for AI scenario analysis. Get one at console.anthropic.com"
    )

    # Try Streamlit secrets first, then manual input
    _fred_secret = st.secrets.get("FRED_API_KEY", "") if hasattr(st, "secrets") else ""
    fred_api_key = _fred_secret or st.text_input(
        "FRED API Key",
        type="password",
        placeholder="abcdef1234567890...",
        help="Free key from https://fredaccount.stlouisfed.org/apikey — required for live macro data"
    )
    if not fred_api_key:
        st.caption("🔑 No FRED key → synthetic macro data will be used.")

    st.markdown('<div class="section-title">📅 DATA RANGE</div>',
                unsafe_allow_html=True)
    data_years = st.slider("Historical Years", 3, 12, 10)
    forecast_months = st.slider("Forecast Horizon (months)", 3, 12, 6)

    st.markdown('<div class="section-title">🤖 ARIMAX ORDER</div>',
                unsafe_allow_html=True)
    p_order = st.selectbox("AR(p)", [1, 2, 3], index=0)
    d_order = st.selectbox("I(d)",  [0, 1], index=1)
    q_order = st.selectbox("MA(q)", [0, 1, 2], index=0)

    st.markdown('<div class="section-title">🔄 LIVE DATA</div>',
                unsafe_allow_html=True)
    use_live = st.toggle("Fetch live USD/INR (Yahoo Finance)", value=True)
    use_fred = st.toggle("Fetch macro from FRED", value=True)

    st.markdown("---")
    if st.button("🔄 Refresh All Data", width='stretch'):
        st.cache_data.clear()
        st.rerun()

    st.caption("Data: Yahoo Finance, FRED. Model: ARIMAX (statsmodels). AI: Claude Sonnet.")


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════

col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<div class="main-header">₹ USD/INR Macro Intelligence</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Live FX · ARIMAX Forecasting · AI Scenario Analysis · Seasonality</div>',
                unsafe_allow_html=True)
with col_h2:
    st.markdown(
        f'<br><span class="live-badge">● LIVE</span>&nbsp; '
        f'<span style="font-family:Space Mono;font-size:0.7rem;color:#64748b;">'
        f'{datetime.today().strftime("%d %b %Y %H:%M")}</span>',
        unsafe_allow_html=True
    )

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

# ── Store fred key in session state so cache functions can read it ──
st.session_state["fred_api_key"] = fred_api_key

with st.spinner("Fetching live data from all sources..."):
    spot = get_live_spot_rate()

    # ── USD/INR (Yahoo Finance) ──
    if use_live:
        usdinr_monthly = load_live_usdinr(data_years)
        usdinr_source  = "Yahoo Finance (live)" if not usdinr_monthly.empty else "Synthetic"
    else:
        idx = pd.date_range("2014-01-01", datetime.today(), freq="MS")
        np.random.seed(7)
        usdinr_monthly = pd.DataFrame(
            {"USD_INR": 63 + np.cumsum(np.random.normal(0.2, 0.5, len(idx)))},
            index=idx
        )
        usdinr_source = "Synthetic"

    # ── FRED macro series (US data) ──
    DATA_SOURCES = {}   # col -> source label
    if use_fred and fred_api_key:
        macro_df, fred_pct = load_macro_fred(
            f"{datetime.today().year - data_years}-01-01", api_key=fred_api_key
        )
        if macro_df is None or fred_pct == 0.0:
            macro_df = build_synthetic_macro(usdinr_monthly.index)
            for c in ["CPI_USA","Crude_Oil","Trade_Balance_India","US_Rate_EFFR"]:
                DATA_SOURCES[c] = "Synthetic"
        else:
            for c in macro_df.columns:
                DATA_SOURCES[c] = "FRED (live)"
            if fred_pct < 1.0:
                synth = build_synthetic_macro(usdinr_monthly.index)
                for col in synth.columns:
                    if col not in macro_df.columns:
                        macro_df[col] = synth[col]
                        DATA_SOURCES[col] = "Synthetic"
    else:
        macro_df = build_synthetic_macro(usdinr_monthly.index)
        for c in macro_df.columns:
            DATA_SOURCES[c] = "Synthetic (no FRED key)"

    # ── RBI Repo Rate ──
    rbi_series, rbi_source = load_rbi_repo_rate()
    DATA_SOURCES["RBI_Repo_Rate"] = rbi_source

    # ── India FX Reserves (FRED RESIRUSD or synthetic) ──
    reserves_series, reserves_source = load_rbi_fx_reserves()
    DATA_SOURCES["Total_Reserves_USD"] = reserves_source

    # ── India CPI (FRED INDCPIALLMINMEI or synthetic) ──
    ind_cpi_series, ind_cpi_source = load_india_cpi_rbi(fred_key=fred_api_key)
    DATA_SOURCES["Ind_CPI"] = ind_cpi_source

    DATA_SOURCES["USD_INR"] = usdinr_source

# ── Merge all sources into one DataFrame ──
combined = usdinr_monthly.copy()

# Merge FRED macro
if macro_df is not None:
    combined = combined.join(macro_df, how="left")

# Override with live RBI Repo Rate if fetched
if rbi_series is not None and rbi_source != "Synthetic":
    rbi_reindexed = rbi_series.reindex(combined.index).ffill().bfill()
    combined["RBI_Repo_Rate"] = rbi_reindexed

# Override with live FX Reserves if fetched
if reserves_series is not None and reserves_source != "Synthetic":
    combined["Total_Reserves_USD"] = reserves_series.reindex(combined.index).ffill().bfill()

# Override with live India CPI if fetched
if ind_cpi_series is not None and ind_cpi_source != "Synthetic":
    combined["Ind_CPI"] = ind_cpi_series.reindex(combined.index).ffill().bfill()

# Fill any still-missing columns with synthetic (never scalar constants — ADF crashes)
_synth = build_synthetic_macro(combined.index)
for _col in ["RBI_Repo_Rate", "Total_Reserves_USD", "Ind_CPI",
             "CPI_USA", "Crude_Oil", "Trade_Balance_India", "US_Rate_EFFR"]:
    if _col not in combined.columns or combined[_col].isna().all():
        combined[_col] = _synth[_col].reindex(combined.index).ffill().bfill()
        DATA_SOURCES[_col] = "Synthetic"

combined = combined.ffill().bfill().dropna()

FEATURE_COLS = [c for c in ["CPI_USA","Crude_Oil","Trade_Balance_India",
                             "US_Rate_EFFR","RBI_Repo_Rate","Total_Reserves_USD","Ind_CPI"]
                if c in combined.columns]

# ── Live Spot Metrics ─────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
spot_rate = spot["rate"] if spot["ok"] else combined["USD_INR"].iloc[-1]
prev_rate = spot["prev"] if spot["ok"] else combined["USD_INR"].iloc[-2]
chg       = spot_rate - prev_rate
pct       = chg / prev_rate * 100

with m1:
    st.markdown(f"""<div class="metric-card">
      <div class="metric-label">USD / INR Spot</div>
      <div class="metric-value">₹{spot_rate:.4f}</div>
      <div class="{'metric-delta-up' if chg < 0 else 'metric-delta-down'}">
        {'▼' if chg < 0 else '▲'} {abs(chg):.4f} ({abs(pct):.2f}%)
      </div></div>""", unsafe_allow_html=True)

with m2:
    ma3 = combined["USD_INR"].rolling(3).mean().iloc[-1]
    st.markdown(f"""<div class="metric-card">
      <div class="metric-label">3-Month MA</div>
      <div class="metric-value">₹{ma3:.4f}</div>
      <div class="metric-delta-{'up' if spot_rate < ma3 else 'down'}">
        vs MA: {'▼' if spot_rate < ma3 else '▲'} {abs(spot_rate - ma3):.4f}
      </div></div>""", unsafe_allow_html=True)

with m3:
    ytd_start = combined.loc[combined.index.year == datetime.today().year, "USD_INR"]
    ytd_val   = ((spot_rate / ytd_start.iloc[0]) - 1) * 100 if not ytd_start.empty else 0
    st.markdown(f"""<div class="metric-card">
      <div class="metric-label">YTD Change</div>
      <div class="metric-value">{ytd_val:+.2f}%</div>
      <div class="{'metric-delta-down' if ytd_val > 0 else 'metric-delta-up'}">
        INR {'weakened' if ytd_val > 0 else 'strengthened'} vs USD YTD
      </div></div>""", unsafe_allow_html=True)

with m4:
    vol = combined["USD_INR"].pct_change().rolling(12).std().iloc[-1] * 100
    st.markdown(f"""<div class="metric-card">
      <div class="metric-label">12M Realised Vol</div>
      <div class="metric-value">{vol:.2f}%</div>
      <div class="metric-delta-{'down' if vol > 3 else 'up'}">
        {'Elevated' if vol > 3 else 'Compressed'} volatility regime
      </div></div>""", unsafe_allow_html=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Live Chart & Forecast",
    "🎭 AI Scenario Analysis",
    "🌊 Seasonality Analysis",
    "🔬 Model Diagnostics",
    "📊 Macro Dashboard",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — LIVE CHART & ARIMAX FORECAST
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">ARIMAX MODEL — FIT & FORECAST</div>',
                unsafe_allow_html=True)

    with st.spinner("Fitting ARIMAX model..."):
        y       = combined["USD_INR"]
        exog    = combined[FEATURE_COLS]
        y_diff  = y.diff().dropna()
        exog_diff = exog.diff().dropna()
        y_diff  = y_diff.loc[exog_diff.index]

        scaler      = StandardScaler()
        exog_scaled = pd.DataFrame(
            scaler.fit_transform(exog_diff),
            columns=exog_diff.columns, index=exog_diff.index
        )
        result = fit_arimax(y_diff, exog_scaled, order=(p_order, d_order, q_order))
        future_exog = exog_scaled.iloc[-forecast_months:]
        forecast_df = forecast_levels(result, future_exog,
                                      last_level=y.iloc[-1],
                                      steps=forecast_months)

    fig1 = plot_usdinr(y, forecast_df)
    st.plotly_chart(fig1, width='stretch')

    # Forecast table
    st.markdown('<div class="section-title">FORECAST VALUES</div>',
                unsafe_allow_html=True)
    fc_display = forecast_df.copy()
    fc_display.index = fc_display.index.strftime("%b %Y")
    fc_display.columns = ["Forecast USD/INR", "Lower 95% CI", "Upper 95% CI"]
    fc_display = fc_display.round(4)
    st.dataframe(fc_display, width='stretch')

    # Train-test backtest
    with st.expander("📉 Backtest Performance (Train/Test Split)"):
        split = "2023-01-01"
        y_tr  = y_diff.loc[:split]; y_te = y_diff.loc[split:]
        ex_tr = exog_scaled.loc[y_tr.index]; ex_te = exog_scaled.loc[y_te.index]

        if len(y_te) >= 3:
            m_tt = SARIMAX(y_tr, exog=ex_tr, order=(p_order, d_order, q_order),
                           enforce_stationarity=False, enforce_invertibility=False)
            r_tt = m_tt.fit(disp=False)
            fc_te = r_tt.get_forecast(steps=len(y_te), exog=ex_te)
            y_pr  = fc_te.predicted_mean
            mae   = mean_absolute_error(y_te, y_pr)
            rmse  = np.sqrt(mean_squared_error(y_te, y_pr))
            last_tr_level = y.loc[y_tr.index[-1]]
            lvl_pred = last_tr_level + y_pr.cumsum()
            lvl_act  = y.loc[y_te.index]
            bc1, bc2, bc3 = st.columns(3)
            bc1.metric("MAE (MoM diff)", f"{mae:.4f}")
            bc2.metric("RMSE (MoM diff)", f"{rmse:.4f}")
            bc3.metric("Test Periods", str(len(y_te)))
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=lvl_act.index, y=lvl_act.values,
                name="Actual", line=dict(color="#f7c948", width=2)))
            fig_bt.add_trace(go.Scatter(x=lvl_pred.index, y=lvl_pred.values,
                name="Predicted", line=dict(color="#f97316", dash="dash", width=2)))
            fig_bt.update_layout(title="Backtest: Actual vs Predicted USD/INR",
                                 **PLOTLY_LAYOUT)
            st.plotly_chart(fig_bt, width='stretch')


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — AI SCENARIO ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">AI-POWERED SCENARIO ANALYSIS</div>',
                unsafe_allow_html=True)
    st.caption(
        "Paste any macro news, RBI/Fed commentary, or geopolitical development. "
        "The AI will extract signals, assign probabilities, and extrapolate the "
        "USD/INR chart under Bull, Base, and Bear scenarios."
    )

    sample_news = (
        "The Federal Reserve signalled a possible pause in rate cuts amid sticky US inflation. "
        "Meanwhile, India's trade deficit widened to $23bn in November due to higher gold imports. "
        "RBI Governor indicated readiness to intervene to prevent excessive rupee depreciation. "
        "Oil prices surged 5% this week on OPEC+ supply cut extension. "
        "Foreign portfolio investors pulled out $2bn from Indian equities."
    )
    news_input = st.text_area(
        "📰 Enter news / macro commentary",
        value=sample_news, height=160,
        placeholder="Paste RBI policy statement, Fed minutes summary, trade data, geopolitical news..."
    )

    col_ai1, col_ai2 = st.columns([1, 2])
    with col_ai1:
        run_ai = st.button("🤖 Run AI Scenario Analysis", type="primary",
                           width='stretch',
                           disabled=(not anthropic_key or not news_input.strip()))
        if not anthropic_key:
            st.warning("Enter Anthropic API key in sidebar to enable AI analysis.")

    if run_ai:
        macro_ctx = {
            "Current_USDINR": round(spot_rate, 4),
            "US_EFFR":        round(float(combined["US_Rate_EFFR"].iloc[-1]),  4) if "US_Rate_EFFR"       in combined else "N/A",
            "RBI_Repo":       round(float(combined["RBI_Repo_Rate"].iloc[-1]), 4) if "RBI_Repo_Rate"       in combined else "N/A",
            "Crude_Oil_USD":  round(float(combined["Crude_Oil"].iloc[-1]),      2) if "Crude_Oil"           in combined else "N/A",
            "FX_Reserves_Bn": round(float(combined["Total_Reserves_USD"].iloc[-1]), 1) if "Total_Reserves_USD" in combined else "N/A",
            "India_CPI_YoY":  round(float(combined["Ind_CPI"].iloc[-1]),        2) if "Ind_CPI"             in combined else "N/A",
            "US_CPI_YoY":     round(float(combined["CPI_USA"].iloc[-1]),         2) if "CPI_USA"             in combined else "N/A",
        }

        with st.spinner("Calling Claude for scenario analysis..."):
            try:
                ai_result = ai_scenario_analysis(
                    news_input, macro_ctx, spot_rate, anthropic_key
                )
            except json.JSONDecodeError as e:
                st.error(f"AI returned invalid JSON: {e}")
                ai_result = None
            except Exception as e:
                st.error(f"AI error: {e}")
                ai_result = None

        if ai_result:
            # ── Summary ───────────────────────────────────────────────────
            st.markdown(f'<div class="ai-response">📌 <b>AI Macro Narrative</b><br><br>'
                        f'{ai_result.get("summary","")}</div>',
                        unsafe_allow_html=True)

            # ── Signals ───────────────────────────────────────────────────
            signals = ai_result.get("signals", [])
            if signals:
                st.markdown('<div class="section-title">KEY SIGNALS DETECTED</div>',
                            unsafe_allow_html=True)
                sig_html = " ".join(
                    f'<span style="background:#1e2d45;color:#94a3b8;border-radius:4px;'
                    f'padding:3px 10px;font-size:0.75rem;margin:2px;display:inline-block;">{s}</span>'
                    for s in signals
                )
                st.markdown(sig_html, unsafe_allow_html=True)

            # ── Scenario Cards ────────────────────────────────────────────
            st.markdown('<div class="section-title">SCENARIO BREAKDOWN</div>',
                        unsafe_allow_html=True)
            sc1, sc2, sc3 = st.columns(3)

            def render_scenario_card(col, key, emoji, pill_class):
                sc = ai_result.get(key, {})
                prob = sc.get("probability_pct", 33)
                with col:
                    st.markdown(
                        f'<span class="scenario-pill {pill_class}">{emoji} {sc.get("label","")}</span>'
                        f'<br><span style="font-family:Space Mono;font-size:1.2rem;color:#f7c948;">'
                        f'{prob}%</span> <span style="font-size:0.75rem;color:#64748b;">probability</span>',
                        unsafe_allow_html=True
                    )
                    st.markdown(f"""
| | 3M | 6M |
|---|---|---|
| **Target** | ₹{sc.get('target_3m', 0):.2f} | ₹{sc.get('target_6m', 0):.2f} |
""")
                    st.caption(sc.get("reasoning", ""))

            render_scenario_card(sc1, "bull", "🟢", "pill-bull")
            render_scenario_card(sc2, "base", "🟣", "pill-base")
            render_scenario_card(sc3, "bear", "🔴", "pill-bear")

            # ── Fan Chart ─────────────────────────────────────────────────
            st.markdown('<div class="section-title">SCENARIO FAN CHART</div>',
                        unsafe_allow_html=True)

            with st.spinner("Building scenario fan chart..."):
                scenarios_fc = {}
                for key in ["bull", "base", "bear"]:
                    try:
                        scenarios_fc[key] = ai_quantify_scenario(
                            key, {}, result, exog_scaled,
                            last_level=y.iloc[-1], steps=forecast_months
                        )
                    except Exception:
                        pass

            if scenarios_fc:
                fig_fan = plot_scenarios(y, scenarios_fc)
                st.plotly_chart(fig_fan, width='stretch')

            # ── Comparison table ──────────────────────────────────────────
            st.markdown('<div class="section-title">SCENARIO COMPARISON TABLE</div>',
                        unsafe_allow_html=True)
            rows = []
            for key, label in [("bull", "🟢 Bull"), ("base", "🟣 Base"), ("bear", "🔴 Bear")]:
                df_s = scenarios_fc.get(key, pd.DataFrame())
                sc   = ai_result.get(key, {})
                rows.append({
                    "Scenario":   label,
                    "Probability":f"{sc.get('probability_pct', 33)}%",
                    "3M Target":  f"₹{sc.get('target_3m', 0):.2f}",
                    "6M Target":  f"₹{sc.get('target_6m', 0):.2f}",
                    "Model 6M":   f"₹{df_s['Forecast'].iloc[-1]:.4f}" if not df_s.empty else "—",
                    "Direction":  sc.get("direction", "—"),
                })
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    else:
        # Demo fan chart (no AI key needed)
        st.markdown('<div class="section-title">DEMO SCENARIO FAN CHART</div>',
                    unsafe_allow_html=True)
        demo = {}
        for key in ["bull", "base", "bear"]:
            try:
                demo[key] = ai_quantify_scenario(key, {}, result, exog_scaled,
                                                  last_level=y.iloc[-1],
                                                  steps=forecast_months)
            except Exception:
                pass
        if demo:
            st.plotly_chart(plot_scenarios(y, demo), width='stretch')
        st.info("💡 Enter your Anthropic API key and paste news to activate AI analysis.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — SEASONALITY
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">SEASONALITY & RETURN DECOMPOSITION</div>',
                unsafe_allow_html=True)
    if len(combined) >= 24:
        st.plotly_chart(plot_seasonality(combined["USD_INR"]),
                        width='stretch')

        # Rolling stats table
        st.markdown('<div class="section-title">MONTHLY RETURN STATISTICS</div>',
                    unsafe_allow_html=True)
        df_m = combined["USD_INR"].to_frame()
        df_m["Return_pct"] = df_m["USD_INR"].pct_change() * 100
        df_m["Month"] = df_m.index.month_name()
        stats = (
            df_m.groupby("Month")["Return_pct"]
            .agg(Avg="mean", Std="std", Min="min", Max="max", Count="count")
            .round(3)
        )
        month_order = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"]
        stats = stats.reindex([m for m in month_order if m in stats.index])
        st.dataframe(stats, width='stretch')
    else:
        st.warning("Need ≥24 months of data for seasonality analysis.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — MODEL DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">ADF STATIONARITY TESTS</div>',
                unsafe_allow_html=True)
    adf_rows = []
    for col in ["USD_INR"] + FEATURE_COLS:
        if col in combined.columns:
            lvl = run_adf(combined[col])
            dff = run_adf(combined[col].diff().dropna())
            def _stat_label(r):
                if r["stationary"] is None: return "—"
                return "✅" if r["stationary"] else "❌"
            adf_rows.append({
                "Variable":         col,
                "Level ADF p-val":  lvl["pval"] if lvl["pval"] is not None else "—",
                "Level Stationary": _stat_label(lvl),
                "Diff ADF p-val":   dff["pval"] if dff["pval"] is not None else "—",
                "Diff Stationary":  _stat_label(dff),
            })
    st.dataframe(pd.DataFrame(adf_rows), width="stretch", hide_index=True)

    st.markdown('<div class="section-title">MODEL SUMMARY</div>',
                unsafe_allow_html=True)
    with st.expander("SARIMAX Summary (click to expand)"):
        st.text(result.summary().as_text())

    st.markdown('<div class="section-title">RESIDUAL DIAGNOSTICS</div>',
                unsafe_allow_html=True)
    st.plotly_chart(plot_residuals(result.resid), width='stretch')

    # Coefficient table — built manually (SARIMAXResults has no summary2())
    st.markdown('<div class="section-title">COEFFICIENT TABLE</div>',
                unsafe_allow_html=True)
    try:
        coeff_df = pd.DataFrame({
            "Coefficient": result.params,
            "Std Error":   result.bse,
            "z-stat":      result.tvalues,
            "p-value":     result.pvalues,
        }).round(4)
        coeff_df["Significant"] = coeff_df["p-value"].apply(
            lambda p: "✅" if p < 0.05 else ("⚠" if p < 0.1 else "❌")
        )
        st.dataframe(coeff_df, width="stretch")
    except Exception as e:
        st.warning(f"Could not build coefficient table: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — MACRO DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-title">MACROECONOMIC INDICATORS</div>',
                unsafe_allow_html=True)

    st.plotly_chart(plot_correlation(
        pd.DataFrame(StandardScaler().fit_transform(combined),
                     columns=combined.columns, index=combined.index)
    ), width='stretch')

    # Individual series charts
    st.markdown('<div class="section-title">TIME SERIES — KEY DRIVERS</div>',
                unsafe_allow_html=True)
    DRIVER_COLORS = {
        "US_Rate_EFFR":       "#60a5fa",
        "RBI_Repo_Rate":      "#a78bfa",
        "Crude_Oil":          "#fb923c",
        "CPI_USA":            "#34d399",
        "Total_Reserves_USD": "#f472b6",
        "Ind_CPI":            "#fbbf24",
    }
    plot_cols = [c for c in DRIVER_COLORS if c in combined.columns]
    if plot_cols:
        n = len(plot_cols)
        rows = (n + 1) // 2
        fig_macro = make_subplots(rows=rows, cols=2,
            subplot_titles=plot_cols, vertical_spacing=0.08)
        for i, col in enumerate(plot_cols):
            r, c = divmod(i, 2)
            fig_macro.add_trace(go.Scatter(
                x=combined.index, y=combined[col],
                name=col, line=dict(color=DRIVER_COLORS.get(col, "#94a3b8"), width=1.5)
            ), row=r + 1, col=c + 1)
        fig_macro.update_layout(height=220 * rows, showlegend=False,
                                **PLOTLY_LAYOUT)
        st.plotly_chart(fig_macro, width='stretch')

    # Raw data download
    st.markdown('<div class="section-title">RAW DATA EXPORT</div>',
                unsafe_allow_html=True)
    csv = combined.to_csv().encode("utf-8")
    st.download_button(
        label="⬇ Download Combined Dataset (CSV)",
        data=csv,
        file_name=f"usd_inr_macro_{datetime.today().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        width='stretch'
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;font-family:Space Mono,monospace;font-size:0.65rem;'
    'color:#334155;letter-spacing:2px;">USD/INR MACRO INTELLIGENCE · ARIMAX + CLAUDE AI · '
    'DATA: YAHOO FINANCE · FRED · NOT FINANCIAL ADVICE</p>',
    unsafe_allow_html=True
)
