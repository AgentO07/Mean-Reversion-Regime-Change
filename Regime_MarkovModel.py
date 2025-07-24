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

def Params(model):
    print("Means of regimes:", model.means_.flatten())
    print("Transition matrix:\n", model.transmat_)
    print("Mean of Regimes: ", model.means_.flatten())
    print("Variances of Regimes: ", np.array([np.diag(cov)[0] for cov in model.covars_]))
    print("Transition matrix:\n", model.transmat_) # prof of going from 1 to 2 and 2 to 1

def plotHMM():
    plt.figure(figsize=(15,5))
    plt.plot(df['DATE'], df['CLOSE'], color='black', lw=1, label='VIX')
    plt.scatter(df['DATE'], df['CLOSE'], c=hidden_states, cmap='coolwarm', s=8, label='Regime')
    plt.title('VIX with HMM-inferred Regimes')
    plt.legend()
    plt.show()



means = model.means_.flatten()
standar_devs = np.sqrt([np.diag(cov)[0] for cov in model.covars_]) 


signals = [] # Store trading signal for each day: 1=long, -1=short, 0=wait

for idx, row in df.iterrows():
    vix = row['CLOSE']
    regime = hidden_states[idx]
    mu = means[regime]
    std = standar_devs[regime]
    long_thresh = mu
    short_thresh = mu + 1.5*std

    if vix < long_thresh:
        signals.append(1)

    elif vix > short_thresh:
        signals.append(-1)
    
    else:
        signals.append(0)

df['signal'] = signals


plt.figure(figsize=(15,6))
plt.plot(df['DATE'], df['CLOSE'], color='gray')
plt.scatter(df['DATE'], df['CLOSE'], c=df['signal'], cmap='bwr', s=12, label='Trading Signal')
plt.title('VIX and Simple Regime-Based Trading Signals (1=long, -1=short, 0=wait)')
plt.show()
        


#Modeling VIX as a regime-switching CIR process (parameters change with regime).
#Writing down the optimal stopping problem:
#At each time, VIX, and regime: is it better to act now or wait?
#Solving coupled partial differential equations (PDEs) that encode this trade/no trade (i.e., “variational inequalities”, see Section 3 of the paper).
#Numerically solve these PDEs using a finite-difference grid.


#Unlike the simple regime thresholds, this approach accounts for mean reversion, random jumps, transaction costs, and the possibility of changing regimes in the future.
#The optimal boundaries it computes say, for every regime and every point in time: "If the VIX is here, what’s the best move—buy, sell, or wait?"
#This approach is dynamically optimal, meaning it fully exploits your right to wait, trade, and learn as the future unfolds.




'''The Process in Detail
Step 1: Set up your model
For each regime, estimate:
CIR parameters: ( \mu_i, \theta_i, \sigma_i )
Transition probabilities/rates (q_{ij}) from your HMM
Decide grid:
Time: ( t_0, t_1, ..., t_N )
VIX: ( s_0, s_1, ..., s_M )
Set transaction costs.
Step 2: Formulate the value functions and the variational inequalities
For each “position” (waiting to enter, hold long, hold short), write down the variational inequality (per Section 2.2–3):
Is the value at a grid point (time, VIX, regime) the max of:
The expected value of waiting (as VIX and regime evolve), or
The value of acting now (entering/exiting, paying cost, etc).
In PDE terms, for each regime, you solve a coupled equation:
The value function for regime i depends not only on itself, but also (through switching rates) on the probability of suddenly jumping to the other regime.
Step 3: Discretize and Iterate
Discretize the PDE using a finite-difference scheme (see Section 3), which produces a system of equations for the value function at each (t, s, i).
For each time step, step backward:
For each VIX grid point, set up the equation relating the value at time (n) to its value at time (n+1)—accounting for drift, volatility, and jumps.
Update your guess for the value (using SOR/PSOR).
Enforce the constraint: Value at each grid point must be $\geq$ the immediate payoff for acting.
Step 4: Find the boundaries
The boundaries are the VIX values at each time where the value function equals the value of acting (buying/selling/closing).
These are the “action frontiers”: if VIX is above/below, act; in-between, wait.
Step 5: Use regimes in trading
At each day in your (real or simulated) time series,
Estimate the current regime (from HMM).
Look up (or interpolate) the corresponding boundary from your FD solution.
Compare the current VIX to the boundary: signal trade if boundary is crossed.
Step 6: Track strategy performance
As you run through your data:
Apply the “trade/no trade” rule at each day based on regime and VIX value.
Keep track of trades, P&L, delays, etc.'''