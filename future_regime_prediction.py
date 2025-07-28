import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
from tqdm import tqdm  # optional, for progress bar

#### LOAD AND PREPARE DATA ####

vix = pd.read_csv('VIX_history.csv')
vix.columns = [c.strip().upper() for c in vix.columns]
vix['DATE'] = pd.to_datetime(vix['DATE'], format='%m/%d/%Y')
vix = vix[['DATE', 'CLOSE']].rename(columns={'CLOSE': 'VIX_CLOSE'})

spx = pd.read_csv('sp500_daily_1990_to_present.csv')
spx.columns = [c.strip().capitalize() for c in spx.columns]
spx['Date'] = pd.to_datetime(spx['Date'], format='%Y-%m-%d')
spx = spx[['Date', 'Close']].rename(columns={'Date': 'DATE', 'Close': 'SPX_CLOSE'})

df = pd.merge(vix, spx, on='DATE', how='inner').sort_values('DATE').reset_index(drop=True)
df['SPX_LOG_RETURN'] = np.log(df['SPX_CLOSE'] / df['SPX_CLOSE'].shift(1))
window = 5
df['SPX_REALIZED_VOL'] = df['SPX_LOG_RETURN'].rolling(window).std()
df_clean = df.dropna(subset=['SPX_LOG_RETURN', 'SPX_REALIZED_VOL']).reset_index(drop=True)

# TRAIN/TEST SPLIT
train_end = pd.Timestamp('2015-12-31')
test_df = df_clean[df_clean['DATE'] > train_end].copy()
test_dates = test_df['DATE'].values

# Walk-forward prediction using only past data up to day t-1
predicted_regimes = []
for t_date in tqdm(test_dates, desc="Walk-forward regime prediction"):
    past_data = df_clean[df_clean['DATE'] < t_date]
    X_past = past_data[['VIX_CLOSE', 'SPX_REALIZED_VOL']].values
    X_today = df_clean[df_clean['DATE'] == t_date][['VIX_CLOSE', 'SPX_REALIZED_VOL']].values

    if len(X_past) < 10:
        predicted_regimes.append(np.nan)  # too little data, skip
        continue

    model = GaussianHMM(n_components=2, covariance_type='full', n_iter=100, random_state=42)
    model.fit(X_past)
    pred = model.predict(X_today)
    predicted_regimes.append(pred[0])

test_df['Regime_Predictive'] = predicted_regimes

# Fit full model on all available data for in-sample comparison
X_full = df_clean[['VIX_CLOSE', 'SPX_REALIZED_VOL']].values
model_full = GaussianHMM(n_components=2, covariance_type='full', n_iter=100, random_state=42)
model_full.fit(X_full)

X_test = test_df[['VIX_CLOSE', 'SPX_REALIZED_VOL']].values
regimes_insample = model_full.predict(X_full[df_clean['DATE'] > train_end])
test_df['Regime_InSample'] = regimes_insample

#### VISUALIZE ####

fig, ax = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

colors = ['royalblue', 'crimson']
labels = ['Regime 0', 'Regime 1']

# Top: Predictive (one-step ahead walk-forward) regimes
for regime in [0, 1]:
    idx = test_df['Regime_Predictive'] == regime
    ax[0].plot(test_df['DATE'][idx], test_df['VIX_CLOSE'][idx], '.', label=f'Predictive {labels[regime]}', color=colors[regime], alpha=0.7)

ax[0].set_ylabel('VIX Close')
ax[0].set_title('VIX Close Predictive Regimes (Walk-forward using only past data)')
ax[0].legend()

# Bottom: In-sample regimes using full model
for regime in [0, 1]:
    idx = test_df['Regime_InSample'] == regime
    ax[1].plot(test_df['DATE'][idx], test_df['VIX_CLOSE'][idx], '.', label=f'In-sample {labels[regime]}', color=colors[regime], alpha=0.5)

ax[1].set_ylabel('VIX Close')
ax[1].set_title('VIX Close In-Sample Regimes (Trained on All Data)')
ax[1].legend()

plt.tight_layout()
plt.show()

#### COMPARE RESULTS ####

# Compare predictive vs in-sample
comparison = test_df['Regime_Predictive'] != test_df['Regime_InSample']
comparison_count = comparison.sum()
print(f"Discrepancies between predictive and in-sample regime predictions: {comparison_count}")
print(test_df[['DATE', 'VIX_CLOSE', 'SPX_REALIZED_VOL', 'Regime_Predictive', 'Regime_InSample']].head(20))
