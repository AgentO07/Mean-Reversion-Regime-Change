import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt

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
train_df = df_clean[df_clean['DATE'] <= train_end].copy()
test_df  = df_clean[df_clean['DATE'] > train_end].copy()

X_train = train_df[['VIX_CLOSE', 'SPX_REALIZED_VOL']].values
X_full  = df_clean[['VIX_CLOSE', 'SPX_REALIZED_VOL']].values

#### TRAIN HMM WITH 3 REGIMES ####

n_regimes = 3

# Out-of-sample model (trained only on training data)
model_train = GaussianHMM(n_components=n_regimes, covariance_type='full', n_iter=100, random_state=42)
model_train.fit(X_train)

# Predict regimes for test set
X_test = test_df[['VIX_CLOSE', 'SPX_REALIZED_VOL']].values
test_regimes_outsample = model_train.predict(X_test)

# Full model (trained on all data)
model_full = GaussianHMM(n_components=n_regimes, covariance_type='full', n_iter=100, random_state=42)
model_full.fit(X_full)
test_regimes_insample = model_full.predict(X_full[df_clean['DATE'] > train_end])

# Assign predictions
test_df['Regime_OutOfSample'] = test_regimes_outsample
test_df['Regime_InSample'] = test_regimes_insample

#### OPTIONAL: LABEL REGIMES BASED ON MEANS ####

# Determine regime meanings (e.g., calm, turbulent, transitional)
# Based on average VIX or SPX realized volatility per regime
means = pd.DataFrame(model_full.means_, columns=['VIX', 'VOL'])
means['Regime'] = means.index

# Sort regimes by VOL and assign labels
means = means.sort_values('VOL').reset_index(drop=True)
regime_labels = {row['Regime']: label for row, label in zip(means.to_dict('records'), ['Calm', 'Changing', 'Turbulent'])}

# Apply label mapping to test_df
test_df['Regime_Out_Label'] = test_df['Regime_OutOfSample'].map(regime_labels)
test_df['Regime_In_Label'] = test_df['Regime_InSample'].map(regime_labels)

#### VISUALIZE ####

fig, ax = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

colors = {'Calm': 'green', 'Changing': 'orange', 'Turbulent': 'red'}
label_order = ['Calm', 'Changing', 'Turbulent']

# Plot Out-of-Sample
for label in label_order:
    idx = test_df['Regime_Out_Label'] == label
    ax[0].plot(test_df['DATE'][idx], test_df['VIX_CLOSE'][idx], '.', label=f'Out-of-sample: {label}', color=colors[label], alpha=0.7)

ax[0].set_ylabel('VIX Close')
ax[0].set_title('VIX Close Out-of-Sample Regimes (3 Regimes)')
ax[0].legend()

# Plot In-Sample
for label in label_order:
    idx = test_df['Regime_In_Label'] == label
    ax[1].plot(test_df['DATE'][idx], test_df['VIX_CLOSE'][idx], '.', label=f'In-sample: {label}', color=colors[label], alpha=0.5)

ax[1].set_ylabel('VIX Close')
ax[1].set_title('VIX Close In-Sample Regimes (3 Regimes)')
ax[1].legend()

plt.tight_layout()
plt.show()

#### COMPARE RESULTS ####

print(test_df[['DATE','VIX_CLOSE','SPX_REALIZED_VOL','Regime_Out_Label','Regime_In_Label']].head(20))

# Count regime mismatches
comparison = test_df['Regime_Out_Label'] != test_df['Regime_In_Label']
print("Discrepancies between in-sample and out-of-sample regime predictions:", comparison.sum())
