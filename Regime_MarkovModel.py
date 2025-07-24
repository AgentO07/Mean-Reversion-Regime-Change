from hmmlearn.hmm import GaussianHMM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('VIX_History.csv', parse_dates=['DATE'])

X = df['CLOSE'].values.reshape(-1,1)#-1 means calc number of rows, 1 means colums, feature

# number of regimes
regimes = 2

#model
model = GaussianHMM(n_components = regimes, covariance_type='full', n_iter=100)
model.fit(X)



# Use model to infer the most likely regime sequence
hidden_states = model.predict(X)  # “hard” regime assignment
probabilities = model.predict_proba(X)    # “soft” probabilities for each regime at each time


print("Means of regimes:", model.means_.flatten())
print("Transition matrix:\n", model.transmat_)


plt.figure(figsize=(15,5))
plt.plot(df['DATE'], df['CLOSE'], color='black', lw=1, label='VIX')
plt.scatter(df['DATE'], df['CLOSE'], c=hidden_states, cmap='coolwarm', s=8, label='Regime')
plt.title('VIX with HMM-inferred Regimes')
plt.legend()
plt.show()