# historical vix data stored in VIX_History.csv

import numpy as np  #math and array handling
import pandas as pd  #reading csv, handling time frame data
from scipy.linalg import solve_banded
# --------- 1. LOAD DATA AND EXTRACT REGIMES -------------

vix_data = pd.read_csv("VIX_History.csv") #loads csv data into pandas dataframe

vix_values = vix_data['CLOSE'].values
dates = pd.to_datetime(vix_data['DATE']) # extracting values



from hmmlearn.hmm import GaussianHMM # for hidden markov models

observations = vix_values.reshape(-1, 1) # HMM needs 2d array


n_regimes = 2
hmm_model = GaussianHMM(n_components=n_regimes, covariance_type="diag", n_iter=1000, random_state=42) # sets maximum EM optimization iterations.
hmm_model.fit(observations) # trains the hmm


regimes = hmm_model.predict(observations) #For every day, this gives us an integer (0 or 1) meaning which regime the model thinks is active.



# --------- 2. CIR PARAMETER ESTIMATION BY REGIME ---------

#theta – the long-term VIX mean for each regime,
#sigma – the volatility (standard deviation) of VIX for each regime,
#kappa – the mean reversion strength (a bit trickier, but we'll do a simple approach).

regime_theta = [17.58, 39.5] #long term mean
regime_sigma = [5.33, 6.42] #stdv vol for vix
regime_kappa = [8.57, 9.0] # mean reversion strength

dt = 1/252  # Assume daily data, 252 trading days per year

# Discount rate (annualized, e.g. 1% per year)
r = 0.05  # <-- SET AT TOP LEVEL, or any value you want to try



# REGIME TRANSITION MATRIX

transition_counts = np.zeros((n_regimes, n_regimes), dtype=int)
for prev, curr in zip(regimes[:-1], regimes[1:]):
    transition_counts[prev, curr] += 1

transition_probs = transition_counts / transition_counts.sum(axis=1, keepdims=True)

#Each row is normalized to sum to 1. For regime i, the row tells you the probability of staying (Q[i,i]) or switching (Q[i,j]) in one day.

Q = np.array([[-0.1,0.1],[0.5,-0.5]])
#Q[i, j] * (f_j - f_i) So if Q[0,1] = 0.5, and the value of future in regime 0
#is 18, in regime 1 is 33, the extra term is: 0.5 * (33 – 18) = 0.5 × 15 = 7.5 interpreted as the effect of potentially jumping to the other regime.



# Building the Discrete Grid for the PDE
c = 0.01     # transaction cost when closing long or opening short
c_hat = 0.01 # transaction cost when opening long or closing short

# Length of contract in years
T = 66/252          # 3 months
N = 120            # 90 time steps (days)
dt = T / N        # time step size/ it is the delta per time step

vix_min, vix_max = 10, 70
M = 200           # 150 grid points for VIX
ds = (vix_max - vix_min) / (M - 1) # vix spacing
vix_grid = np.linspace(vix_min, vix_max, M)
time_grid = np.linspace(0, T, N+1)

#Initialize storage for futures contract prices at each grid point (all regimes):

futures = np.zeros((n_regimes, N+1, M))

#utures[i, t, s] is the value of the future with t steps to expiry, VIX at level s, in regime i.


for i in range(n_regimes):
    futures[i, -1, :] = vix_grid  # payoff at expiry

    #At expiry, the value of the contract is simply the VIX itself.



#We need to compute, for each regime, at each time step, a tridiagonal linear system corresponding to the finite-difference discretization of the CIR PDE (with regime switching).



#Build the Crank-Nicolson System (PDE Step) for Each Regime


#This collects the previously estimated CIR parameters for use in the finite-difference equations.



### __________________REFER TO STEP 6 IN DOCS FOR DETAILED EXPLANATION _______


    # ...we'll use these for the PDE coefficients

#Step 6.2: For every interior VIX grid point, compute finite difference PDE coefficients


def crank_nicolson_pde_step(futures, Q, r, dt, ds, vix_grid, regime_kappa, regime_theta, regime_sigma, t):
    """
    Perform one backward time step for all regimes: 
    updates futures[:,t,:] in place by solving the regime-coupled CIR+switching PDE.
    
    Parameters:
    futures      - shape (n_regimes, N+1, M): current solution at all future times; gets updated for t
    Q            - (n_regimes, n_regimes): regime switching generator matrix
    r            - float: risk-free discount rate
    dt           - float: time step size
    ds           - float: VIX grid spacing
    vix_grid     - (M,): grid of VIX values across states
    regime_kappa, regime_theta, regime_sigma - arrays of length n_regimes: parameters per regime
    t            - integer: current time step (going backwards)
    """
    n_regimes = len(regime_kappa)
    M = len(vix_grid)
    for i in range(n_regimes):
        kappa = regime_kappa[i]
        theta = regime_theta[i]
        sigma = regime_sigma[i]
        
        # Prepare the A matrix and rhs vector
        A = np.zeros((M, M))
        rhs = np.zeros(M)
        
        # --- Boundary conditions ---
        # At min VIX
        A[0, 0] = 1
        rhs[0] = vix_grid[0]
        
        # At max VIX
        A[M-1, M-1] = 1
        rhs[M-1] = vix_grid[M-1]
        
        # --- Interior points (loop over S) ---
        for j in range(1, M-1):
            S = vix_grid[j]
            # Finite-diff Crank-Nicolson coefficients
            # a = 0.25 * dt * (sigma^2 * S / ds^2 - kappa*(theta - S) / ds)
            # b = -0.5 * dt * (sigma^2 * S / ds^2 + r)
            # c = 0.25 * dt * (sigma^2 * S / ds^2 + kappa*(theta - S) / ds)
            sigma2S = sigma * sigma * S
            a = 0.25 * dt * (sigma2S / ds**2 - kappa * (theta - S) / ds)
            b = -0.5 * dt * (sigma2S / ds**2 + r)
            c = 0.25 * dt * (sigma2S / ds**2 + kappa * (theta - S) / ds)
            
            # Fill in the matrix A (for "new" time step, t)
            A[j, j-1] = -a
            A[j, j]   = 1 - b
            A[j, j+1] = -c
            
            # --- Build the right-hand side (involving only "old" values at t+1) ---
            # Regime coupling (forward Euler: explicit)
            coupling = 0.0
            for k in range(n_regimes):
                if k != i:
                    # Q[i,k] is the per-time-step regime jump rate
                    coupling += Q[i, k] * dt * (futures[k, t+1, j] - futures[i, t+1, j])
            
            rhs[j] = (
                a * futures[i, t+1, j-1]
                + (1 + b) * futures[i, t+1, j]
                + c * futures[i, t+1, j+1]
                + coupling
            )
        
        # --- Solve the tridiagonal system A x_new = rhs ---
        # Prepare matrix for scipy's solve_banded
        ab = np.zeros((3, M))
        ab[0, 1:] = np.diagonal(A, 1)   # upper diag
        ab[1, :] = np.diagonal(A, 0)    # main diag
        ab[2, :-1] = np.diagonal(A, -1) # lower diag
        
        # Solve for this regime, this time step, all VIX values
        futures[i, t, :] = solve_banded((1, 1), ab, rhs)




for t in reversed(range(N)):  # N is number of interior time steps (not including expiry)
    crank_nicolson_pde_step(
        futures, Q, r, dt, ds, vix_grid, 
        regime_kappa, regime_theta, regime_sigma, t
    )


V = np.zeros_like(futures)  # same shape: (n_regimes, N+1, M)
#V will hold, for every (regime, time, VIX), the optimal liquidation value if you already hold a long position.

V[:, -1, :] = futures[:, -1, :] - c  # sell for future value minus transaction cost


#Now, backward induction for each earlier timestep:
#We'll go backwards in time, deciding for every cell:

#"Should I exit now at current price?"
#or "Should I wait another day, hoping for a better exit?"


for t in reversed(range(N)):
    for i in range(n_regimes):
        for s in range(M):
            # 1. Value if we exit now (sell): get the current futures value minus cost
            exit_now = futures[i, t, s] - c
            
            # 2. Value if we wait (keep the position): discount tomorrow's value, average over possible regime jumps
            wait = 0
            for j in range(n_regimes):
                # Probability of moving from regime i to j in one timestep
                prob = Q[i, j] * dt if i != j else 1 + Q[i, i] * dt
                wait += prob * V[j, t+1, s]  # Note: V[j, t+1, s] is value tomorrow in regime j, same S

            wait *= np.exp(-r * dt)  # discount for one time step

            # 3. Optimal value: whichever is better
            V[i, t, s] = max(exit_now, wait)


J = np.zeros_like(futures)  # shape: (n_regimes, N+1, M)
J[:, -1, :] = 0


for t in reversed(range(N)):
    for i in range(n_regimes):
        for s in range(M):
            # 1. If you enter NOW (buy long): pay futures[i,t,s]+c_hat, get the value of exiting later (V)
            entry_value = V[i, t, s] - (futures[i, t, s] + c_hat)
            entry_value = max(entry_value, 0)  # never negative: don’t enter if not profitable

            # 2. Or, you wait ONE day: expected, discounted future value (over possible regime jumps)
            wait = 0
            for j in range(n_regimes):
                prob = Q[i, j] * dt if i != j else 1 + Q[i, i] * dt
                wait += prob * J[j, t+1, s]
            wait *= np.exp(-r * dt)

            # 3. Optimal: maximum of entering now or waiting
            J[i, t, s] = max(entry_value, wait)

U = np.zeros_like(futures)
U[:, -1, :] = futures[:, -1, :] + c_hat  # at expiry/buy to cover at settlment
for t in reversed(range(N)):
    for i in range(n_regimes):
        for s in range(M):
            exit_now = futures[i, t, s] + c_hat
            wait = 0
            for j in range(n_regimes):
                prob = Q[i,j] * dt if i != j else 1 + Q[i,i] * dt
                wait += prob * U[j, t+1, s]
            wait *= np.exp(-r*dt)
            U[i,t,s] = min(exit_now, wait) # minimize, because losses are bad in short trades


K = np.zeros_like(futures)
K[:, -1, :] = 0  # can't enter at expiry

for t in reversed(range(N)):
    for i in range(n_regimes):
        for s in range(M):
            entry_value = (futures[i, t, s] - c) - U[i, t, s]
            entry_value = max(entry_value, 0)
            wait = 0
            for j in range(n_regimes):
                prob = Q[i,j] * dt if i != j else 1 + Q[i,i] * dt
                wait += prob * K[j, t+1, s]
            wait *= np.exp(-r*dt)
            K[i,t,s] = max(entry_value, wait)


P = np.zeros_like(futures)
P[:, -1, :] = 0 # can't enter at expiry

for t in reversed(range(N)):
    for i in range(n_regimes):
        for s in range(M):
            entry_value = max(J[i,t,s], K[i,t,s])  # best of both entries
            wait = 0
            for j in range(n_regimes):
                prob = Q[i,j] * dt if i != j else 1 + Q[i,i] * dt
                wait += prob * P[j, t+1, s]
            wait *= np.exp(-r*dt)
            P[i,t,s] = max(entry_value, wait)

entry_long_boundary = np.full((n_regimes, N+1), np.nan)
for i in range(n_regimes):
    for t in range(N+1):
        entry_now_values = (V[i, t, :] - (futures[i, t, :] + c_hat)).clip(min=0)
        diff = J[i, t, :] - entry_now_values
        # Find first crossing from negative to non-negative
        crossings = np.where((diff[:-1] < 0) & (diff[1:] >= 0))[0]
        if crossings.size > 0:
            entry_long_boundary[i, t] = vix_grid[crossings[0] + 1]
        else:
            entry_long_boundary[i, t] = np.nan


exit_long_boundary = np.full((n_regimes, N+1), np.nan)
for i in range(n_regimes):
    for t in range(N+1):
        exit_now_values = futures[i, t, :] - c
        boundary_indices = np.where(np.isclose(V[i, t, :], exit_now_values, atol=1e-6))[0]
        if len(boundary_indices) > 0:
            exit_long_boundary[i, t] = vix_grid[boundary_indices.min()]

entry_short_boundary = np.full((n_regimes, N+1), np.nan)
exit_short_boundary = np.full((n_regimes, N+1), np.nan)
for i in range(n_regimes):
    for t in range(N+1):
        entry_now_values = (futures[i, t, :] - c) - U[i, t, :]
        boundary_indices = np.where(np.isclose(K[i, t, :], entry_now_values.clip(min=0), atol=1e-6))[0]
        if len(boundary_indices) > 0:
            entry_short_boundary[i, t] = vix_grid[boundary_indices.max()]  # max: high-VIX for shorting

        exit_now_values = futures[i, t, :] + c_hat
        boundary_indices = np.where(np.isclose(U[i, t, :], exit_now_values, atol=1e-6))[0]
        if len(boundary_indices) > 0:
            exit_short_boundary[i, t] = vix_grid[boundary_indices.max()]    # max: high-VIX for short exit


mid = N//2
print("Long entry at mid, regime 0:")
print("VIX grid:", vix_grid)
print("J:", J[0, mid, :])
print("entry_now:", (V[0, mid, :] - (futures[0, mid, :] + c_hat)).clip(min=0))


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

labels = ['Regime 0 (calm)', 'Regime 1 (volatile)']
colors = ['blue', 'red']

plt.plot(vix_grid, J[0, mid, :], label='J (value to enter long)')
plt.plot(vix_grid, (V[0, mid, :] - (futures[0, mid, :] + c_hat)).clip(min=0), label='entry_now')
plt.legend(); plt.xlabel('VIX'); plt.ylabel('Value'); plt.title('Entry Value Curve')
plt.show()


for i in range(n_regimes):
    plt.plot(time_grid, entry_long_boundary[i, :], label=f'Long Entry, {labels[i]}', color=colors[i], linestyle='-')
    plt.plot(time_grid, exit_long_boundary[i, :], label=f'Long Exit, {labels[i]}', color=colors[i], linestyle='--')
    plt.plot(time_grid, entry_short_boundary[i, :], label=f'Short Entry, {labels[i]}', color=colors[i], linestyle='-.')
    plt.plot(time_grid, exit_short_boundary[i, :], label=f'Short Exit, {labels[i]}', color=colors[i], linestyle=':')

plt.xlabel('Time to expiry (years)')
plt.ylabel('VIX Level')
plt.title('Optimal Trading Boundaries (by regime and time)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()