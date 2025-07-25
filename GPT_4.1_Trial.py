#############################
# 1. Data Import & HMM Regime Detection
#############################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from scipy.linalg import solve_banded

# -- Step 1: Load and Prepare Data
df = pd.read_csv('VIX_History.csv', parse_dates=['DATE'])
VIX = df['CLOSE'].values.reshape(-1,1)

# -- Step 2: Fit a two-regime HMM
regimes = 2
model = GaussianHMM(n_components=regimes, covariance_type="full", n_iter=100)
model.fit(VIX)
hidden_states = model.predict(VIX)
probas = model.predict_proba(VIX)

# -- Diagnostics: Plot VIX colored by regime
plt.figure(figsize=(13,4))
plt.plot(df['DATE'], df['CLOSE'], color='k', lw=1, label='VIX')
plt.scatter(df['DATE'], df['CLOSE'], c=hidden_states, cmap='coolwarm', s=8, label='Regime')
plt.title('VIX with inferred regimes (via HMM)')
plt.xlabel('Date'); plt.ylabel('VIX'); plt.legend(); plt.show()

# -- Print HMM parameter
print("Regime means:", model.means_.flatten())
print("Variances:", [np.diag(cov)[0] for cov in model.covars_])
print("Transition Matrix:\n", model.transmat_)

#############################
# 2. Parameter Extraction per Regime
#############################
means = model.means_.flatten()     # regime means, to use as CIR theta
stds = np.sqrt([np.diag(cov)[0] for cov in model.covars_])
P = model.transmat_                # Discrete transition matrix (not continuous-time, but OK here)

# We'll make some basic CIR parameter guesses:
theta_array = means
sigma_array = stds
mu_array = np.full(regimes, 8.0) # Assume mean-reversion speed is 8.0 for illustration (you'd calibrate this in practice)
r = 0.05                         # Discount rate

#############################
# 3. Finite Difference Grid Setup
#############################
T = 22                           # total days (roughly 1 month to expiry)
N = T * 2                        # finer time grid
dt = T / N

Smin, Smax = 10, 60              # VIX range
M = 100
ds = (Smax - Smin) / M
time_grid = np.linspace(0, T, N+1)
vix_grid = np.linspace(Smin, Smax, M+1)

num_regimes = regimes
V = np.zeros((num_regimes, M+1, N+1)) # "Value function" grid

cost_exit = 1.0  # Example transaction cost

#############################
# 4. Terminal Condition (at expiry)
#############################
# At expiry, the payoff to liquidating (closing) is just immediate value less cost
for reg in range(num_regimes):
    for m in range(M+1):
        V[reg, m, N] = vix_grid[m] - cost_exit

#############################
# 5. Constructing Matrix Bands
#############################
def get_alpha_beta_gamma(mu, th, s, sig, dt, ds, r, M):
    phi = mu * (th - s)
    alpha =  dt/(4*ds) * (sig**2/ds - phi)
    beta  = -dt/2 * (r + sig**2 / ds**2)
    gamma = dt/(4*ds) * (sig**2/ds + phi)
    return alpha, beta, gamma

#############################
# 6. Backward Induction (Dynamic Programming)
#############################
for n in reversed(range(1, N+1)):                   # Time-stepping backwards
    immediate_payoff = vix_grid - cost_exit         # What you get by stopping now

    for reg in range(num_regimes):
        mu = mu_array[reg]
        th = theta_array[reg]
        sig = sigma_array[reg]
        alpha, beta, gamma = get_alpha_beta_gamma(mu, th, vix_grid, sig, dt, ds, r, M)
        
        # Set up the tridiagonal matrix bands for solve_banded
        lower = -alpha[1:]                  # M values
        main = 1 - beta                     # M+1 values
        upper = -gamma[:-1]                 # M values
        ab = np.zeros((3, M+1))
        ab[0,1:] = upper
        ab[1,: ] = main
        ab[2,:-1] = lower

        # Right-hand side (add regime switching term: P[reg, other]*V[other])
        rhs = np.zeros(M+1)
        for m in range(M+1):
            rhs[m] = 0
            for reg2 in range(num_regimes):
                if reg2 != reg:
                    rhs[m] += dt * P[reg, reg2] * V[reg2, m, n] # Use transition * future value in other regime
            rhs[m] += V[reg, m, n]  # Use current regime's value

        # Apply boundaries if needed (could also pad rhs with Dirichlet)
        # Solve for interior points
        V[reg, :, n-1] = solve_banded((1,1), ab, rhs)
        # Project onto constraint (optimal stopping!): cannot be below immediate payoff
        V[reg, :, n-1] = np.maximum(V[reg, :, n-1], immediate_payoff)

#############################
# 7. Extract Optimal Boundaries
#############################
optimal_boundaries = np.zeros((num_regimes, N+1))
immediate_payoff = vix_grid - cost_exit # again

for reg in range(num_regimes):
    for n in range(N+1):
        # Find first index where value function equals immediate payoff (should be a boundary)
        boundary = np.argmax(V[reg, :, n] <= immediate_payoff)
        optimal_boundaries[reg, n] = vix_grid[boundary]

#############################
# 8. Plot Result
#############################
plt.figure(figsize=(10,6))
for reg in range(num_regimes):
    plt.plot(time_grid, optimal_boundaries[reg, :], label=f'Regime {reg+1}')
plt.xlabel('Time to expiry (days)')
plt.ylabel('Optimal boundary (VIX)')
plt.title('Optimal Exercise Boundaries for VIX Futures (by regime)')
plt.legend()
plt.gca().invert_xaxis()
plt.show()