# RUPICAST
AI-powered USD/INR macro forecasting dashboard. Combines live FX data (Yahoo Finance/FRED), ARIMAX time-series modelling, and Claude AI to deliver real-time forecasts, Bull/Base/Bear scenario fan charts, and seasonality analysis — all in a Streamlit interface.

📈 Tab 1 — Live Chart & Forecast

Fetches daily USD/INR from Yahoo Finance (auto-resampled to monthly), with a 1-hour cache
Fetches macro data from FRED (US EFFR, CPI, Crude Oil, Trade Balance) — falls back to synthetic data if FRED is blocked
Fits your ARIMAX model with configurable p,d,q from the sidebar
Plots the full historical series + forecast with 95% confidence intervals
Built-in train/test backtest with MAE and RMSE

🎭 Tab 2 — AI Scenario Analysis ⭐

Paste any news (RBI minutes, Fed commentary, geopolitical news, trade data)
Claude reads it, extracts macro signals, writes a narrative, and assigns Bull / Base / Bear probabilities
Each scenario has AI-generated 3M and 6M price targets + reasoning
The ARIMAX model is re-run with macro shocks for each scenario (e.g. Bull = -25bps EFFR, -$5 crude, +$10bn reserves)
Outputs a fan chart showing all three scenario paths

🌊 Tab 3 — Seasonality Analysis

Monthly seasonal deviation bar chart (which months INR typically weakens/strengthens)
Year-on-year heatmap of monthly returns
Return distribution histogram
STL trend decomposition

🔬 Tab 4 — Model Diagnostics

ADF stationarity tests for all variables (levels + first diff)
Full SARIMAX summary table
Residual time-series + histogram
Coefficient table with p-values

📊 Tab 5 — Macro Dashboard

Full correlation heatmap (scaled)
Individual time-series charts for all 6 macro drivers
CSV export of the combined dataset
