import numpy as np
import pandas as pd
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt

# ===================== 1. DATA AND HMM =====================
vix_data = pd.read_csv("VIX_History.csv")
vix_values = vix_data['CLOSE'].values
dates = pd.to_datetime(vix_data['DATE'])
observations = vix_values.reshape(-1, 1)

from hmmlearn.hmm import GaussianHMM
n_regimes = 2
hmm_model = GaussianHMM(n_components=n_regimes, covariance_type="diag", n_iter=1000, random_state=42)
hmm_model.fit(observations)
regimes = hmm_model.predict(observations)

# ===================== 2. (OPTIONAL) PARAMETER FITTING =====================
# --- OVERRIDE FOR DEMO: use realistic, test-friendly CIR and Q params
regime_sigma = [5.33, 6.42]
regime_theta = [17.58, 29.5]
regime_kappa = [8.57, 9.0]
Q = np.array([[-0.1, 0.1], [0.5, -0.5]])
c = c_hat = 0.01  # zero cost for diagnostic, can add later

# # # --- AUTOMATIC ESTIMATION (uncomment if needed for real-data fit) ----
# dt = 1/252
# regime_theta, regime_sigma, regime_kappa = [], [], []
# for i in range(n_regimes):
#     idx = (regimes == i)
#     S = vix_values[idx]
#     theta = np.mean(S)
#     sigma = np.std(np.diff(S)) / np.sqrt(np.mean(S))
#     if len(S) > 1:
#         dS = np.diff(S)
#         S_lag = S[:-1]
#         X = (theta - S_lag) * dt
#         Y = dS
#         kappa = np.sum(X*Y) / np.sum(X*X)
#     else:
#         kappa = 1.0
#     regime_theta.append(theta)
#     regime_sigma.append(sigma)
#     regime_kappa.append(max(kappa, 1e-3))

# # Regime transition matrix (may not match manual Q above!)
# transition_counts = np.zeros((n_regimes, n_regimes), dtype=int)
# for prev, curr in zip(regimes[:-1], regimes[1:]):
#     transition_counts[prev, curr] += 1
# with np.errstate(divide='ignore', invalid='ignore'):
#     transition_probs = np.nan_to_num(transition_counts / transition_counts.sum(axis=1, keepdims=True))
# Q = np.zeros((n_regimes, n_regimes))
# for i in range(n_regimes):
#     for j in range(n_regimes):
#         if i != j:
#             Q[i, j] = transition_probs[i, j] / dt
#     Q[i, i] = -np.sum(Q[i, :]) + Q[i, i]

# ===================== 3. BUILD PDE GRID =====================
T = 66/252
N = 120
dt = T/N
vix_min, vix_max = 5, 60
M = 200
ds = (vix_max - vix_min)/(M-1)
vix_grid = np.linspace(vix_min, vix_max, M)
time_grid = np.linspace(0, T, N+1)
r = 0.05

futures = np.zeros((n_regimes, N+1, M))
for i in range(n_regimes):
    futures[i, -1, :] = vix_grid

# ===================== 4. CIR + REGIME SWITCHING PDE SOLVER =====================
def crank_nicolson_pde_step(futures, Q, r, dt, ds, vix_grid, regime_kappa, regime_theta, regime_sigma, t):
    n_regimes = len(regime_kappa)
    M = len(vix_grid)
    for i in range(n_regimes):
        kappa = regime_kappa[i]
        theta = regime_theta[i]
        sigma = regime_sigma[i]
        A = np.zeros((M, M))
        rhs = np.zeros(M)
        A[0, 0] = 1; rhs[0] = vix_grid[0]
        A[-1, -1] = 1; rhs[-1] = vix_grid[-1]
        for j in range(1, M-1):
            S = vix_grid[j]
            sigma2S = sigma**2 * S
            a = 0.25*dt*(sigma2S/ds**2 - kappa*(theta-S)/ds)
            b = -0.5*dt*(sigma2S/ds**2 + r)
            c = 0.25*dt*(sigma2S/ds**2 + kappa*(theta-S)/ds)
            A[j, j-1] = -a
            A[j, j] = 1 - b
            A[j, j+1] = -c
            # regime coupling (explicit)
            coupling = np.sum([Q[i, k]*dt*(futures[k, t+1, j]-futures[i, t+1, j]) for k in range(n_regimes) if k!=i])
            rhs[j] = (a*futures[i, t+1, j-1] + (1+b)*futures[i, t+1, j] + c*futures[i, t+1, j+1] + coupling)
        ab = np.zeros((3, M))
        ab[0, 1:] = np.diagonal(A, 1)
        ab[1, :] = np.diagonal(A, 0)
        ab[2, :-1] = np.diagonal(A, -1)
        futures[i, t, :] = solve_banded((1, 1), ab, rhs)

for t in reversed(range(N)):
    crank_nicolson_pde_step(futures, Q, r, dt, ds, vix_grid, regime_kappa, regime_theta, regime_sigma, t)

# ===================== 5. OPTIMAL STOPPING / VALUE FUNCTIONS =====================
V = np.zeros_like(futures); J = np.zeros_like(futures)
U = np.zeros_like(futures); K = np.zeros_like(futures)
P = np.zeros_like(futures)
V[:, -1, :] = futures[:, -1, :] - c
J[:, -1, :] = 0
U[:, -1, :] = futures[:, -1, :] + c_hat
K[:, -1, :] = 0
P[:, -1, :] = 0

for t in reversed(range(N)):
    for i in range(n_regimes):
        for s in range(M):
            # --- V: exit long
            exit_now = futures[i, t, s] - c
            wait_v = np.exp(-r * dt) * np.sum([(Q[i, j]*dt if i!=j else 1+Q[i,i]*dt)*V[j, t+1, s] for j in range(n_regimes)])
            V[i, t, s] = max(exit_now, wait_v)
            # --- J: entry long
            entry_value = max(V[i, t, s] - (futures[i, t, s] + c_hat), 0)
            wait_j = np.exp(-r * dt) * np.sum([(Q[i, j]*dt if i!=j else 1+Q[i,i]*dt)*J[j, t+1, s] for j in range(n_regimes)])
            J[i, t, s] = max(entry_value, wait_j)
            # --- U: exit short
            exit_now = futures[i, t, s] + c_hat
            wait_u = np.exp(-r * dt) * np.sum([(Q[i, j]*dt if i!=j else 1+Q[i,i]*dt)*U[j, t+1, s] for j in range(n_regimes)])
            U[i, t, s] = min(exit_now, wait_u)
            # --- K: entry short
            entry_value = max((futures[i, t, s]-c) - U[i, t, s], 0)
            wait_k = np.exp(-r * dt) * np.sum([(Q[i, j]*dt if i!=j else 1+Q[i,i]*dt)*K[j, t+1, s] for j in range(n_regimes)])
            K[i, t, s] = max(entry_value, wait_k)
            # --- P: entry either
            entry_value = max(J[i, t, s], K[i, t, s])
            wait_p = np.exp(-r * dt) * np.sum([(Q[i, j]*dt if i!=j else 1+Q[i,i]*dt)*P[j, t+1, s] for j in range(n_regimes)])
            P[i, t, s] = max(entry_value, wait_p)

# ===================== 6. EXTRACT BOUNDARIES VIA CROSSING =====================
entry_long_boundary = np.full((n_regimes, N+1), np.nan)
exit_long_boundary = np.full((n_regimes, N+1), np.nan)
entry_short_boundary = np.full((n_regimes, N+1), np.nan)
exit_short_boundary = np.full((n_regimes, N+1), np.nan)

for i in range(n_regimes):
    for t in range(N+1):
        # Long entry: first crossing J - entry_now from negative to non-negative
        entry_now_vals = (V[i, t, :] - (futures[i, t, :] + c_hat)).clip(min=0)
        diff = J[i, t, :] - entry_now_vals
        crossings = np.where((diff[:-1]<0)&(diff[1:]>=0))[0]
        if crossings.size > 0:
            entry_long_boundary[i, t] = vix_grid[crossings[0]+1]
        # Long exit: first V == exit_now
        exit_now_vals = futures[i, t, :] - c
        mask = np.isclose(V[i, t, :], exit_now_vals, atol=1e-6)
        idx = np.where(mask)[0]
        if len(idx) > 0:
            exit_long_boundary[i, t] = vix_grid[idx[0]]
        # Short entry:
        entry_now_vals = (futures[i, t, :] - c) - U[i, t, :]
        diff = K[i, t, :] - entry_now_vals.clip(min=0)
        crossings = np.where((diff[:-1]<0)&(diff[1:]>=0))[0]
        if crossings.size > 0:
            entry_short_boundary[i, t] = vix_grid[crossings[-1]]
        # Short exit:
        exit_now_vals = futures[i, t, :] + c_hat
        mask = np.isclose(U[i, t, :], exit_now_vals, atol=1e-6)
        idx = np.where(mask)[0]
        if len(idx) > 0:
            exit_short_boundary[i, t] = vix_grid[idx[-1]]

# ===================== 7. PLOTS =====================
mid = N // 2
plt.figure(figsize=(12, 6))
plt.plot(vix_grid, J[0, mid, :], label='J (value to enter long)')
plt.plot(vix_grid, (V[0, mid, :] - (futures[0, mid, :] + c_hat)).clip(min=0), label='entry_now')
plt.legend(); plt.xlabel('VIX'); plt.ylabel('Value'); plt.title('Entry Value Curve')
plt.show()

labels = ['Regime 0 (calm)', 'Regime 1 (volatile)']
colors = ['blue', 'red']
plt.figure(figsize=(12, 7))
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


plt.figure(figsize=(8,6))
t_in_days = np.arange(N+1)
plt.plot(t_in_days, exit_short_boundary[0, :], 'b--', label=r'$U_1$')
plt.plot(t_in_days, entry_short_boundary[0, :], 'b-',  label=r'$K_1$')
plt.plot(t_in_days, exit_short_boundary[1, :], 'r--', label=r'$U_2$')
plt.plot(t_in_days, entry_short_boundary[1, :], 'r-',  label=r'$K_2$')
plt.xlabel('time (days)')
plt.ylabel('VIX')
plt.legend()
plt.show()