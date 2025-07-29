import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt

#### LOAD AND PREPARE DATA ####

# Load VIX data
vix = pd.read_csv('VIX_history.csv')
vix.columns = [c.strip().upper() for c in vix.columns]  # Clean column names
vix['DATE'] = pd.to_datetime(vix['DATE'], format='%m/%d/%Y')  # Convert date column to datetime
vix = vix[['DATE', 'CLOSE']].rename(columns={'CLOSE': 'VIX_CLOSE'})  # Keep relevant columns, rename CLOSE to VIX_CLOSE

# Load S&P 500 (SPX) data
spx = pd.read_csv('sp500_daily_1990_to_present.csv')
spx.columns = [c.strip().capitalize() for c in spx.columns]  # Clean column names
spx['Date'] = pd.to_datetime(spx['Date'], format='%Y-%m-%d')  # Convert date column to datetime
spx = spx[['Date', 'Close']].rename(columns={'Date': 'DATE', 'Close': 'SPX_CLOSE'})  # Keep relevant columns

# Merge VIX and SPX data on DATE, sort chronologically
df = pd.merge(vix, spx, on='DATE', how='inner').sort_values('DATE').reset_index(drop=True)

# Compute SPX log returns
df['SPX_LOG_RETURN'] = np.log(df['SPX_CLOSE'] / df['SPX_CLOSE'].shift(1))

# Compute 5-day rolling standard deviation (realized volatility)
window = 5
df['SPX_REALIZED_VOL'] = df['SPX_LOG_RETURN'].rolling(window).std()

# Drop rows with NaNs from rolling window or first return
df_clean = df.dropna(subset=['SPX_LOG_RETURN', 'SPX_REALIZED_VOL']).reset_index(drop=True)

# Define train/test split at end of 2015
train_end = pd.Timestamp('2020-12-31')
train_df = df_clean[df_clean['DATE'] <= train_end].copy()  # Train data: 1990 - 2015
test_df  = df_clean[df_clean['DATE'] > train_end].copy()   # Test data: 2016 onward

# Extract training features (VIX + realized volatility)
X_train = train_df[['VIX_CLOSE', 'SPX_REALIZED_VOL']].values
X_full  = df_clean[['VIX_CLOSE', 'SPX_REALIZED_VOL']].values  # For full-data training later



# Train Gaussian HMM on training data (2 hidden regimes)
model_train = GaussianHMM(n_components=3, covariance_type='full', n_iter=150, random_state=42)
model_train.fit(X_train)

# Predict hidden regimes on test data using model trained only on 1990–2015
X_test = test_df[['VIX_CLOSE', 'SPX_REALIZED_VOL']].values
test_regimes_outsample = model_train.predict(X_test)

# Train another HMM on full dataset (for in-sample prediction)
model_full = GaussianHMM(n_components=3, covariance_type='full', n_iter=150, random_state=42)
model_full.fit(X_full)

# Predict regimes again (in-sample), on the same test period
test_regimes_insample = model_full.predict(X_full[df_clean['DATE'] > train_end])

#### MAP REGIMES TO CONSISTENT ORDER BY MEAN VIX ####
# Assign predicted regimes to DataFrame
test_df['Regime_OutOfSample'] = test_regimes_outsample
test_df['Regime_InSample'] = test_regimes_insample

# Remap regime labels by mean VIX
def remap_regimes_by_mean_vix(df_subset, regime_col_name):
    regime_stats = df_subset.groupby(regime_col_name)['VIX_CLOSE'].mean()
    sorted_regimes = regime_stats.sort_values().index.tolist()
    mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_regimes)}
    return df_subset[regime_col_name].map(mapping)

# Now apply the remapping
test_df['Regime_OutOfSample'] = remap_regimes_by_mean_vix(test_df, 'Regime_OutOfSample')
test_df['Regime_InSample'] = remap_regimes_by_mean_vix(test_df, 'Regime_InSample')


#### VISUALIZE ####

# Create 2 subplots (for Out-of-Sample and In-Sample regime visualizations)
fig, ax = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Define color scheme and labels for the 3 regimes
colors = ['royalblue', 'crimson', 'darkorange']
labels = ['Regime 0', 'Regime 1', 'Regime 2']

# OUT-OF-SAMPLE: plot VIX Close values colored by regime (from model trained only on train data)
for regime in [0, 1, 2]:
    idx = test_df['Regime_OutOfSample'] == regime
    ax[0].plot(test_df['DATE'][idx], test_df['VIX_CLOSE'][idx], '.', 
               label=f'Out-of-sample {labels[regime]}', color=colors[regime], alpha=0.7)

ax[0].set_ylabel('VIX Close')
ax[0].set_title(f'VIX Close Out-of-Sample Regimes (Trained 1990-{train_end})')
ax[0].legend()

# IN-SAMPLE: plot VIX Close values colored by regime (from model trained on all data)
for regime in [0, 1, 2]:
    idx = test_df['Regime_InSample'] == regime
    ax[1].plot(test_df['DATE'][idx], test_df['VIX_CLOSE'][idx], '.', 
               label=f'In-sample {labels[regime]}', color=colors[regime], alpha=0.7)

ax[1].set_ylabel('VIX Close')
ax[1].set_title('VIX Close In-Sample Regimes (Trained on All Data)')
ax[1].legend()

plt.tight_layout()
plt.show()

#### COMPARE RESULTS ####

# Print side-by-side comparison of regime labels for first 20 rows of test data
print(test_df[['DATE','VIX_CLOSE','SPX_REALIZED_VOL','Regime_OutOfSample','Regime_InSample']].head(20))

# Count how many days the two regime labels (in-sample vs out-of-sample) disagree
comparison = test_df['Regime_OutOfSample'] != test_df['Regime_InSample']
print("Discrepancies between in-sample and out-of-sample regime predictions:", comparison.sum())



# Example for Out-of-Sample regimes
print(test_df.groupby('Regime_OutOfSample')[['VIX_CLOSE', 'SPX_REALIZED_VOL']].mean())

# Example for In-Sample regimes
print(test_df.groupby('Regime_InSample')[['VIX_CLOSE', 'SPX_REALIZED_VOL']].mean())
