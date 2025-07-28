import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from scipy.linalg import solve_banded

# --------- 1. LOAD DATA AND EXTRACT REGIMES -------------
df = pd.read_csv('VIX_History.csv', parse_dates=['DATE'])
VIX = df['CLOSE'].values.reshape(-1, 1)
regimes = 2

model = GaussianHMM(n_components=regimes, covariance_type="full", n_iter=100)
model.fit(VIX)
hidden_states = model.predict(VIX)
print("Regime sizes:", [np.sum(hidden_states == i) for i in range(regimes)])

# --------- 2. CIR PARAMETER ESTIMATION BY REGIME ---------
def cir_ols_params(series):
    series = np.maximum(series, 3)
    Y = np.diff(series)
    X = series[:-1]
    dt = 1.0
    if len(X) < 10 or np.all(X == X[0]):
        return 0.2, np.median(series), 1.2  # fallback: realistic μ/σ for vol
    A = np.vstack([np.ones(len(X)), -X]).T
    coeff, _, _, _ = np.linalg.lstsq(A, Y/dt, rcond=None)
    mu_est = np.clip(coeff[1], 0.15, 1.5)
    theta_est = np.clip(-coeff[0]/mu_est, 10, 40)
    # median series for θ if OLS is crazy
    if theta_est < 8 or theta_est > 50:
        theta_est = np.median(series)
    sigma_est = np.clip(np.std(Y / np.sqrt(np.maximum(X, 1))), 0.5, 2.5)
    return mu_est, theta_est, sigma_est


params_by_regime = []
for reg in range(regimes):
    idx = np.where(hidden_states == reg)[0]
    series = df['CLOSE'].iloc[idx].values
    mu, theta, sigma = cir_ols_params(series)
    params_by_regime.append([mu, theta, sigma])
    print(f"Regime {reg}: μ={mu:.3f}, θ={theta:.2f}, σ={sigma:.3f}, size={len(series)}")

mu_array = np.array([x[0] for x in params_by_regime])
theta_array = np.array([x[1] for x in params_by_regime])
sigma_array = np.array([x[2] for x in params_by_regime])

# --------- 3. Q-GENERATOR: BY RUN-LENGTH (NOT LOG P) -------
# Calculate average run length in each regime, set Q
R = []
for reg in range(regimes):
    idx = np.where(hidden_states == reg)[0]
    diff = np.diff(idx)
    runs = np.split(idx, np.where(diff != 1)[0] + 1)
    if len(runs) > 0:
        R.append(np.mean([len(r) for r in runs]))
    else:
        R.append(len(idx) if len(idx) > 0 else 1)
Q = np.zeros((regimes, regimes))
for i in range(regimes):
    for j in range(regimes):
        if i != j:
            Q[i, j] = 1.0/R[i]
    Q[i, i] = -1.0/R[i]
print("\nGenerator Q (estimated):\n", Q)

# --------- 4. SETUP FINITE DIFFERENCE GRIDS -----------
T = 200       # days
N = T*2
dt = T / N
Smin, Smax = 10, 80
M = 100
ds = (Smax - Smin) / M
time_grid = np.linspace(0, T, N+1)
vix_grid = np.linspace(Smin, Smax, M+1)
cost_exit = 0
cost_entry = 0
r = 0.05

num_regimes = regimes
V = np.zeros((num_regimes, M+1, N+1)) # Exit value function
J = np.zeros((num_regimes, M+1, N+1)) # Entry value function

# --------- 5. TERMINAL CONDITION -----------
for reg in range(num_regimes):
    for m in range(M+1):
        V[reg, m, N] = vix_grid[m] - cost_exit
        J[reg, m, N] = 0  # can't enter at expiry

# --------- 6. DOUBLE STOPPING, REGIME-COUPLED PDE SOLVE -----------

def get_coeffs(mu, th, s, sig, dt, ds, r):
    phi = mu * (th - s)
    alpha = dt/(4*ds) * (sig**2/ds - phi)
    beta = -dt/2 * (r + sig**2 / ds**2)
    gamma = dt/(4*ds) * (sig**2/ds + phi)
    return alpha, beta, gamma

for n in reversed(range(1, N+1)):
    immediate_exit = vix_grid - cost_exit
    for reg in range(num_regimes):
        mu, th, sig = mu_array[reg], theta_array[reg], sigma_array[reg]
        alpha, beta, gamma = get_coeffs(mu, th, vix_grid, sig, dt, ds, r)
        lower = -alpha[1:]
        main = 1 - beta
        upper = -gamma[:-1]
        ab = np.zeros((3, M+1))
        ab[0,1:] = upper
        ab[1,:] = main
        ab[2,:-1] = lower

        # ------ EXIT -----------
        rhsV = np.zeros(M+1)
        for m in range(M+1):
            rhsV[m] = V[reg, m, n]
            for reg2 in range(num_regimes):
                if reg2 != reg:
                    rhsV[m] += dt * Q[reg, reg2] * V[reg2, m, n]
        Vsol = solve_banded((1,1), ab, rhsV)
        V[reg, :, n-1] = np.maximum(Vsol, immediate_exit)

    # ------ ENTRY (only after exit has been updated) ------
    for reg in range(num_regimes):
        mu, th, sig = mu_array[reg], theta_array[reg], sigma_array[reg]
        alpha, beta, gamma = get_coeffs(mu, th, vix_grid, sig, dt, ds, r)
        lower = -alpha[1:]
        main = 1 - beta
        upper = -gamma[:-1]
        ab = np.zeros((3, M+1))
        ab[0,1:] = upper
        ab[1,:] = main
        ab[2,:-1] = lower

        # REWARD for entry: (V - (S+cost_entry))+   (skip "+" in code, it's already max'd)
        immediate_entry = np.maximum(V[reg, :, n-1] - (vix_grid + cost_entry), 0)
        rhsJ = np.zeros(M+1)
        for m in range(M+1):
            rhsJ[m] = J[reg, m, n]
            for reg2 in range(num_regimes):
                if reg2 != reg:
                    rhsJ[m] += dt * Q[reg, reg2] * J[reg2, m, n]
        Jsol = solve_banded((1,1), ab, rhsJ)
        J[reg, :, n-1] = np.maximum(Jsol, immediate_entry)

# --------- 7. REGION MAP: BUY/WAIT/SELL ---------------
region_map = np.zeros((num_regimes, M+1, N+1), dtype=int)
for reg in range(num_regimes):
    for n in range(N+1):
        for m in range(M+1):
            # BUY: when entering (J > 0)
            if J[reg, m, n] > 0 and (J[reg, m, n] == V[reg, m, n] - vix_grid[m] - cost_entry):
                region_map[reg, m, n] = 1
            # SELL: when optimal to exit (V == immediate_exit)
            elif np.abs(V[reg, m, n] - (vix_grid[m] - cost_exit)) < 1e-6:
                region_map[reg, m, n] = -1
            else:
                region_map[reg, m, n] = 0  # WAIT

# --------- 8. PLOT: REGION MAP (as in paper figs) --------------
fig, axs = plt.subplots(num_regimes, 1, figsize=(11,7), sharex=True)
for reg in range(num_regimes):
    # Regions: y=VIX, x=time (reversed axis so t=0 is right as in paper)
    im = axs[reg].imshow(region_map[reg,:,:], aspect='auto', cmap='bwr',
                         extent=[T, 0, Smin, Smax], origin='lower',
                         vmin=-1, vmax=1)
    axs[reg].set_title(f'Regime {reg+1}: Optimal Trading/Entry Regions')
    axs[reg].set_ylabel('VIX')
    cbar = plt.colorbar(im, ax=axs[reg], orientation='vertical', ticks=[-1,0,1])
    cbar.ax.set_yticklabels(['SELL','WAIT','BUY'])
axs[-1].set_xlabel('Time to Expiry (days)')
plt.tight_layout()
plt.show()


'''for i in range(n_regimes):
    idx = (regimes == i) #idx gives us the days in that regime,
    S = vix_values[idx] #S is the sequence of VIX values during those days.

    theta = np.mean(S)
    sigma = np.std(np.diff(S)) / np.sqrt(np.mean(S))  # crude CIR volatility estimate

    # refer to Discretizing for Estimation in the paper


    if len(S) > 1:
        dS = np.diff(S)
        S_lag = S[:-1]
        # Simple OLS to fit dS = kappa*(theta - S_lag)*dt
        X = (theta - S_lag) * dt
        Y = dS
        kappa = np.sum(X*Y) / np.sum(X*X)
    else:
        kappa = 1.0  # fallback if not enough data

    regime_theta.append(theta)
    regime_sigma.append(sigma)
    regime_kappa.append(max(kappa, 1e-3))  # enforce kappa > 0

    print(f"Regime {i}: theta={regime_theta[i]:.2f}, sigma={regime_sigma[i]:.2f}, kappa={regime_kappa[i]:.2f}")'''