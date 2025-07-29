import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt

#### LOAD AND PREPARE DATA ####

# Load your data as shown earlier
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
train_df = df_clean[df_clean['DATE'] <= train_end].copy()
test_df  = df_clean[df_clean['DATE'] > train_end].copy()

X_train = train_df[['VIX_CLOSE', 'SPX_REALIZED_VOL']].values
X_full  = df_clean[['VIX_CLOSE', 'SPX_REALIZED_VOL']].values

# Train on 1990-2015
model_train = GaussianHMM(n_components=2, covariance_type='full', n_iter=100, random_state=42)
model_train.fit(X_train)

# Predict regimes for test set (out-of-sample)
X_test = test_df[['VIX_CLOSE', 'SPX_REALIZED_VOL']].values
test_regimes_outsample = model_train.predict(X_test)

# Train on ALL data (for in-sample comparison)
model_full = GaussianHMM(n_components=2, covariance_type='full', n_iter=100, random_state=42)
model_full.fit(X_full)
test_regimes_insample = model_full.predict(X_full[df_clean['DATE'] > train_end])

# Add regimes to the test dataframe for comparison
test_df['Regime_OutOfSample'] = test_regimes_outsample
test_df['Regime_InSample'] = test_regimes_insample

#### VISUALIZE ####

fig, ax = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

colors = ['royalblue', 'crimson']
labels = ['Regime 0', 'Regime 1']

# Plot VIX w/color by regime - Out-of-Sample
for regime in [0, 1]:
    idx = test_df['Regime_OutOfSample'] == regime
    ax[0].plot(test_df['DATE'][idx], test_df['VIX_CLOSE'][idx], '.', label=f'Out-of-sample {labels[regime]}', color=colors[regime], alpha=0.7)

ax[0].set_ylabel('VIX Close')
ax[0].set_title('VIX Close Out-of-Sample Regimes (Trained 1990-2015)')
ax[0].legend()

# Plot VIX w/color by regime - In-Sample
for regime in [0, 1]:
    idx = test_df['Regime_InSample'] == regime
    ax[1].plot(test_df['DATE'][idx], test_df['VIX_CLOSE'][idx], '.', label=f'In-sample {labels[regime]}', color=colors[regime], alpha=0.5)

ax[1].set_ylabel('VIX Close')
ax[1].set_title('VIX Close In-Sample Regimes (Trained on All Data)')
ax[1].legend()

plt.tight_layout()
plt.show()

#### COMPARE RESULTS ####

# Quick comparison table for the first 20 test days
print(test_df[['DATE','VIX_CLOSE','SPX_REALIZED_VOL','Regime_OutOfSample','Regime_InSample']].head(20))

# Compare actual differences
comparison = test_df['Regime_OutOfSample'] != test_df['Regime_InSample']
print("Discrepancies between in-sample and out-of-sample regime predictions:", comparison.sum())


#### SCATTERPLOT COMPARISON ####
'''
fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)

# Scatterplot: Out-of-sample regimes
for regime in [0, 1]:
    idx = test_df['Regime_OutOfSample'] == regime
    ax[0].scatter(test_df['VIX_CLOSE'][idx], test_df['SPX_REALIZED_VOL'][idx], 
                  color=colors[regime], label=labels[regime], alpha=0.6)
ax[0].set_title('Out-of-Sample Regime Classification')
ax[0].set_xlabel('VIX Close')
ax[0].set_ylabel('SPX Realized Volatility')
ax[0].legend()

# Scatterplot: In-sample regimes
for regime in [0, 1]:
    idx = test_df['Regime_InSample'] == regime
    ax[1].scatter(test_df['VIX_CLOSE'][idx], test_df['SPX_REALIZED_VOL'][idx], 
                  color=colors[regime], label=labels[regime], alpha=0.4)
ax[1].set_title('In-Sample Regime Classification')
ax[1].set_xlabel('VIX Close')
ax[1].legend()

plt.suptitle('Scatterplot of Regimes by VIX vs. Realized Volatility', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
'''