from hmmlearn.hmm import GaussianHMM # type: ignore
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded



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
    short_thresh = mu + 1*std

    if vix < long_thresh:
        signals.append(1)

    elif vix > short_thresh:
        signals.append(-1)
    
    else:
        signals.append(0)

df['signal'] = signals

def only():
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






theta_array = means
sigma_array = standar_devs



# Time grid: say, model 30 trading days until expiry

T = 30          # total days until expiry
N = T*2            # number of time steps (2 per day: finer grid)
dt = T / N            # time increment, in days

# VIX grid: let's cover VIX from 10 to 80
Smin = 10
Smax = 80
M = 140             # number of VIX steps (about 0.5 VIX per grid step)
ds = (Smax - Smin) / M


# Time grid: N+1 evenly spaced time points from 0 to T
time_grid = np.linspace(0, T, N+1)

# VIX grid: M+1 evenly spaced VIX points from Smin to Smax
vix_grid = np.linspace(Smin, Smax, M+1)


num_regimes = 2
# For each regime, build (M+1, N+1) array: VIX (rows) x Time (columns)
V = np.zeros((num_regimes, M+1, N+1))

#Here: V[0, :, :] is the value function grid for regime 1, V[1, :, :] is for regime 2.


alpha = np.zeros((num_regimes, M+1))
beta = np.zeros((num_regimes, M+1))
gamma = np.zeros((num_regimes, M+1))
r = 0.08 # Example discount rate (annualized risk-free rate, fine for most FD models)



# transition matrix: rows sum to 1
P = model.transmat_  # shape: (2, 2)
q = -np.log(np.diag(P))   # crude approximation for continuous-time rates

    
mu_array = np.array([5.0,1.0])
cost_exit = 0.2 # or whatever your transaction cost is
for _regime in range(num_regimes):
    for m in range(M+1):
        s = vix_grid[m]
        V[_regime, m, N] = s - cost_exit

for n in reversed(range(1, N+1)):
    for reg in range(num_regimes):
        
        mu = mu_array[reg]    # mean reversion for this regime
        th = theta_array[reg]      # long-run mean for this regime
        sig = sigma_array[reg]     # vol for this regime

        phi = mu * (th - vix_grid)
        alpha_row = (dt/(4*ds)) * (sig**2/ds - phi)
        beta_row  = -(dt/2) * (r + sig**2 / ds**2)
        gamma_row = (dt/(4*ds)) * (sig**2/ds + phi)
        # Set up banded matrix for tridiagonal solver
        lower = -alpha_row[1:]          # lower diagonal (M values)
        main  = 1 - beta_row            # main diagonal (M+1 values)
        upper = -gamma_row[:-1]         # upper diagonal (M values)       # upper diagonal (M values)

        ab = np.zeros((3, M+1))
        ab[0,1:] = upper
        ab[1,: ] = main
        ab[2,:-1] = lower

        # Right-hand side
        # (includes future value + regime-coupling; make sure to adjust for boundaries at m=0, m=M)
        rhs = np.zeros(M+1)
        for m in range(M+1):
            # Start with current value at time n
            vval = V[reg, m, n]
            # Add regime coupling (sum over j != reg)
            for other in range(num_regimes):
                if other != reg:
                    # Qij = transition rate from reg to other
                    rhs[m] += dt * P[reg, other] * V[other, m, n]
            # Complete with explicit terms (see paper Eq 3.3+)
            rhs[m] += alpha[reg, m] * V[reg, m-1, n] if m>0 else 0
            rhs[m] += (1+beta[reg, m])*V[reg, m, n]
            rhs[m] += gamma[reg, m] * V[reg, m+1, n] if m<M else 0

        # Solve tridiagonal system for V[reg, :, n-1]
        V[reg, :, n-1] = solve_banded((1,1), ab, rhs)
        V[reg, :, n-1] = np.maximum(V[reg, :, n-1], vix_grid - cost_exit)

        # Enforce optimal stopping (projected SOR – simp


        # 1. Compute/assemble alpha, beta, gamma arrays for this regime (shape M+1)
        # 2. Build the tridiagonal matrix for the finite difference method
        # 3. At every VIX grid point, include regime-coupling terms (from other regime's value)
        # 4. Solve linear system for V[:, n-1, reg]
        # 5. For each m in 0..M:
        #    V[reg, m, n-1] = max(V[reg, m, n-1], immediate_payoff_at_that_point)

optimal_boundaries = np.zeros((num_regimes, N+1))

immediate_payoff = vix_grid - cost_exit

for reg in range(num_regimes):
    for n in range(N+1):
        # Find the boundary—where value first falls to immediate payoff as we increase VIX
        boundary_idx = None
        for m in range(M+1):
            if V[reg, m, n] <= immediate_payoff[m]:  # Exercise region
                boundary_idx = m
                break
        if boundary_idx is None:
            boundary_idx = M  # If never optimal, set to max VIX

        optimal_boundaries[reg, n] = vix_grid[boundary_idx]


plt.figure(figsize=(10,6))
for reg in range(num_regimes):
    plt.plot(time_grid, optimal_boundaries[reg,:], label=f'Regime {reg+1}')
plt.xlabel('Time to expiry (days)')
plt.ylabel('Optimal exercise VIX boundary')
plt.title('Optimal Exercise Boundaries (by regime)')
plt.legend()
plt.gca().invert_xaxis()  # So time moves left to right (now to expiry)
plt.show()