import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM

# 1. Load VIX data
vix = pd.read_csv('VIX_history.csv')
# Standardize column names
vix.columns = [c.strip().upper() for c in vix.columns]
vix['DATE'] = pd.to_datetime(vix['DATE'], format='%m/%d/%Y')  # US month-first
vix = vix[['DATE', 'CLOSE']].rename(columns={'CLOSE': 'VIX_CLOSE'})

# 2. Load S&P data
spx = pd.read_csv('sp500_daily_1990_to_present.csv')
spx.columns = [c.strip().capitalize() for c in spx.columns]
spx['Date'] = pd.to_datetime(spx['Date'], format='%Y-%m-%d')
spx = spx[['Date', 'Close']].rename(columns={'Date': 'DATE', 'Close': 'SPX_CLOSE'})

# 3. Merge/join on 'DATE'
df = pd.merge(vix, spx, on='DATE', how='inner').sort_values('DATE').reset_index(drop=True)

# 4. Calculate S&P log returns
df['SPX_LOG_RETURN'] = np.log(df['SPX_CLOSE'] / df['SPX_CLOSE'].shift(1))

# 5. Calculate rolling 5-day realized volatility (standard deviation of log returns)
window = 5
df['SPX_REALIZED_VOL'] = df['SPX_LOG_RETURN'].rolling(window).std()

# 6. Drop initial rows with NaNs (due to rolling window and shift)
df_clean = df.dropna(subset=['SPX_LOG_RETURN', 'SPX_REALIZED_VOL']).reset_index(drop=True)

# 7. Prepare 2D feature input
X = df_clean[['VIX_CLOSE', 'SPX_REALIZED_VOL']].values  # shape: [num_days, 2]

# 8. Fit HMM
model = GaussianHMM(n_components=2, covariance_type='full', n_iter=100)
model.fit(X)

# 9. Regime prediction
regimes = model.predict(X)

# 10. Add regime to dataframe and optionally save
df_clean['Regime'] = regimes

df_clean.to_csv('vix_spx_hmm_regime.csv', index=False)

# -------- OPTIONAL: Inspect/plot ------------
import matplotlib.pyplot as plt


# --- Plot regimes with two y-axes ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Color map for regimes
colors = plt.get_cmap('cool')(df_clean['Regime'] / df_clean['Regime'].max())

# Plot VIX
ax1.plot(df_clean['DATE'], df_clean['VIX_CLOSE'], label='VIX Close', color='tab:blue')
ax1.scatter(df_clean['DATE'], df_clean['VIX_CLOSE'], c=colors, s=10, label='Regime')
ax1.set_ylabel('VIX Close')
ax1.set_title('VIX Close with HMM Regimes')
ax1.legend()

# Plot Realized Vol
ax2.plot(df_clean['DATE'], df_clean['SPX_REALIZED_VOL'], label='S&P Realized Vol (5-day, log returns)', color='tab:orange')
ax2.scatter(df_clean['DATE'], df_clean['SPX_REALIZED_VOL'], c=colors, s=10, label='Regime')
ax2.set_ylabel('S&P Realized Volatility')
ax2.set_title('S&P Realized Vol (5-day, log returns) with HMM Regimes')
ax2.legend()

plt.xlabel('Date')
plt.tight_layout()
plt.show()
