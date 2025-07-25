import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM # type: ignore
from scipy.optimize import minimize
from scipy.linalg import logm
from scipy.special import i0 as I0
from scipy.linalg import solve_banded

# ===== 1. Load, fit HMM to VIX =====
df = pd.read_csv("VIX_History.csv", parse_dates=['DATE'])
df = df.sort_values('DATE')
vix = df['Close'].values.reshape(-1, 1)
dates = df['DATE'].values

print("Fitting 2-regime HMM...")
hmm = GaussianHMM(n_components=2, covariance_type="full", n_iter=500, random_state=42)
hmm.fit(vix)
regimes = hmm.predict(vix)
state_probs = hmm.predict_proba(vix)
df['Regime'] = regimes

# Plot
plt.figure(figsize=(14,5))
for i in range(2):
    plt.plot(df['DATE'][regimes==i], df['Close'][regimes==i], '.', label=f"Regime {i+1}")
plt.legend(); plt.ylabel('VIX Close'); plt.title('VIX with HMM Regimes')
plt.show()

theta_hmm = hmm.means_.flatten()
sigma_hmm = np.sqrt([np.diag(cov)[0] for cov in hmm.covars_])
P = hmm.transmat_

print("Means (theta):", theta_hmm)
print("Stddevs (sigma):", sigma_hmm)
print("Transition matrix (daily):\n", P)

# ===== 2. Fit CIR per regime =====
def cir_neg_log_likelihood_vec(params, x, dt):
    mu, theta, sigma = params
    if mu <= 0 or theta <= 0 or sigma <= 0:
        return 1e20
    x1 = x[:-1]
    x2 = x[1:]
    c = (2*mu)/(sigma**2*(1-np.exp(-mu*dt)))
    q = 2*mu*theta/(sigma**2)-1
    u = c*np.exp(-mu*dt)*x1
    v = c*x2
    mask = (x1 > 0) & (x2 > 0)
    pdf = np.zeros_like(u)
    pdf[mask] = (c * np.exp(-(u[mask]+v[mask]))
                 * ((v[mask]/u[mask])**(q/2))
                 * I0(2*np.sqrt(u[mask]*v[mask])))
    pdf[pdf<=0] = 1e-20
    return -np.sum(np.log(pdf + 1e-20))

dt = 1/252
cir_params = []
print("Fitting CIR per regime...")
for i in range(2):
    vixr = df.loc[df['Regime'] == i, 'Close'].values
    if len(vixr) < 5:
        cir_params.append((np.nan, np.nan, np.nan))
        continue
    guess = [1, np.mean(vixr), np.std(vixr)]
    bounds = ((1e-4, 10), (0.1, 100), (1e-4, 20))
    res = minimize(cir_neg_log_likelihood_vec, guess, args=(vixr, dt), bounds=bounds)
    mu, theta, sigma = res.x
    cir_params.append((mu, theta, sigma))
    print(f"Regime {i}: mu={mu:.3f}, theta={theta:.3f}, sigma={sigma:.3f}")

mu_hat = np.array([cir_params[0][0], cir_params[1][0]])
theta_hat = np.array([cir_params[0][1], cir_params[1][1]])
sigma_hat = np.array([cir_params[0][2], cir_params[1][2]])
for i in range(2):
    if np.isnan(mu_hat[i]):
        print(f"Regime {i+1} too short; using HMM mean/std.")
        mu_hat[i] = 1.0
        theta_hat[i] = theta_hmm[i]
        sigma_hat[i] = sigma_hmm[i]

# ===== 3. Generator Matrix Q (per year) =====
Q = np.real(logm(P))
Q_peryear = Q / (1/252)
print("\n--- Parameter Summary ---")
print("mu =", mu_hat)
print("theta =", theta_hat)
print("sigma =", sigma_hat)
print("Q (per year) =\n", np.round(Q_peryear,4))
print("-------------------------")


# ===== 4. Optimal Trading Boundaries: PDE/VI Solver =====

# Use your estimates
mu = mu_hat
theta = theta_hat
sigma = sigma_hat
q = Q_peryear

r = 0.05
Smax = 60.0
T = 22/252
Tmat = 66/252
c = 0.01

NS = 140
NT = 70
dS = Smax/(NS-1)
dt_pde = T/NT
S_grid = np.linspace(0, Smax, NS)
t_grid = np.linspace(0, T, NT+1)

# --- Futures price PDE under regime switching CIR ---
def price_futures_CIR_CN(S_grid, t_grid, mu_q, theta_q, sigma_q, q, Tmat):
    NS, NT, Nreg = len(S_grid), len(t_grid)-1, 2
    f = np.zeros((Nreg, NT+1, NS))
    f[:, -1, :] = S_grid
    dt = t_grid[1] - t_grid[0]
    dS = S_grid[1] - S_grid[0]
    for n in reversed(range(NT)):
        for i in range(Nreg):
            a = np.zeros(NS)
            b = np.zeros(NS)
            c_ = np.zeros(NS)
            d = np.zeros(NS)
            for m in range(NS):
                S = S_grid[m]
                if m == 0 or m == NS-1:
                    b[m] = 1.0
                    d[m] = S
                else:
                    sig2 = 0.5 * sigma_q[i]**2 * S
                    mu_term = mu_q[i]*(theta_q[i] - S)
                    a[m] = -(dt/(4*dS)) * (sig2/dS - mu_term)
                    b[m] = 1 + dt * (r + (sig2/dS**2))
                    c_[m] = -(dt/(4*dS)) * (sig2/dS + mu_term)
                    d[m] = (1 - dt * (r + (sig2/dS**2))) * f[i, n+1, m] \
                        + (dt/(4*dS)) * (sig2/dS - mu_term) * f[i, n+1, m-1] \
                        + (dt/(4*dS)) * (sig2/dS + mu_term) * f[i, n+1, m+1]
                    for j in range(Nreg):
                        if j != i:
                            d[m] += dt * q[i, j] * f[j, n+1, m]
            ab = np.zeros((3, NS))
            ab[0, 1:] = c_[1:]
            ab[1, :] = b
            ab[2, :-1] = a[:-1]
            f[i, n, :] = solve_banded((1,1), ab, d)
    return f

futures_grid = price_futures_CIR_CN(S_grid, t_grid, mu, theta, sigma, q, Tmat)

# --- PSOR for variational inequality ---
def solve_V_J_CN_PSOR(S_grid, t_grid, mu, theta, sigma, q, r, c, futures_grid, omega=1.2, tol=1e-6, max_iter=300):
    NS = len(S_grid)
    NT = len(t_grid)-1
    Nreg = 2
    V = np.zeros((Nreg, NT+1, NS))
    J = np.zeros((Nreg, NT+1, NS))
    for i in range(Nreg):
        V[i, -1, :] = futures_grid[i, -1, :] - c
        J[i, -1, :] = np.maximum(V[i, -1, :] - (futures_grid[i, -1, :] + c), 0)
    for n in reversed(range(NT)):
        for i in range(Nreg):
            # V
            vnext = V[i, n+1, :]
            fnxt = futures_grid[i, n, :]
            a, b, c_ = np.zeros(NS), np.zeros(NS), np.zeros(NS)
            rhs = np.zeros(NS)
            for m in range(NS):
                S = S_grid[m]
                if m == 0 or m == NS-1:
                    b[m] = 1.0
                    rhs[m] = fnxt[m] - c
                else:
                    sig2 = 0.5 * sigma[i]**2 * S
                    mu_term = mu[i]*(theta[i] - S)
                    a[m] = -(dt_pde/(4*dS))*(sig2/dS - mu_term)
                    b[m] = 1 + dt_pde*(r + (sig2/dS**2))
                    c_[m]= -(dt_pde/(4*dS))*(sig2/dS + mu_term)
                    rhs[m] = (1 - dt_pde*(r + (sig2/dS**2)))*vnext[m] \
                        + (dt_pde/(4*dS)) * (sig2/dS - mu_term)*vnext[m-1] \
                        + (dt_pde/(4*dS)) * (sig2/dS + mu_term)*vnext[m+1]
                    for j in range(Nreg):
                        if j!=i:
                            rhs[m] += dt_pde*q[i, j]*V[j, n+1, m]
            v_old = V[i, n, :].copy()
            for it in range(max_iter):
                v_new = v_old.copy()
                for m in range(NS):
                    if m == 0 or m==NS-1: 
                        v_new[m] = rhs[m]
                    else:
                        y = (rhs[m] - a[m]*v_new[m-1] - c_[m]*v_new[m+1])/b[m]
                        v_star = v_old[m] + omega*(y-v_old[m])
                        v_new[m] = max(fnxt[m]-c, v_star)
                if np.max(np.abs(v_new-v_old)) < tol: break
                v_old = v_new
            V[i, n, :] = v_new
            # J
            payoff = np.maximum(V[i, n, :] - (fnxt + c), 0)
            jnext = J[i, n+1, :]
            rhs2 = np.zeros(NS)
            for m in range(NS):
                if m==0 or m==NS-1:
                    b[m]=1.0
                    rhs2[m]=payoff[m]
                else:
                    rhs2[m] = (1 - dt_pde*(r + (sig2/dS**2)))*jnext[m] \
                              + (dt_pde/(4*dS))*(sig2/dS - mu_term)*jnext[m-1]\
                              + (dt_pde/(4*dS))*(sig2/dS + mu_term)*jnext[m+1]
                    for j in range(Nreg):
                        if j!=i:
                            rhs2[m] += dt_pde*q[i,j]*J[j,n+1,m]
            j_old = J[i, n, :].copy()
            for it in range(max_iter):
                j_new = j_old.copy()
                for m in range(NS):
                    if m == 0 or m==NS-1:
                        j_new[m]=rhs2[m]
                    else:
                        y = (rhs2[m] - a[m]*j_new[m-1] - c_[m]*j_new[m+1])/b[m]
                        j_star = j_old[m] + omega*(y - j_old[m])
                        j_new[m]=max(payoff[m], j_star)
                if np.max(np.abs(j_new-j_old)) < tol: break
                j_old = j_new
            J[i, n, :] = j_new
    return V, J

print("Solving for boundaries (can take 1-2 minutes if grid is fine)...")
V, J = solve_V_J_CN_PSOR(S_grid, t_grid, mu, theta, sigma, q, r, c, futures_grid)

# --- Extract boundary where VI active/equals payoff (first crossing of exercise) ---
def find_boundary(S_grid, value, exercise):
    boundary = np.full(value.shape[0], np.nan)
    for n in range(value.shape[0]):
        d = value[n] - exercise[n]
        idxs = np.where(d <= 0)[0]
        if len(idxs) > 0:
            boundary[n] = S_grid[idxs[0]]
        else:
            # No crossing -- plot the minimum |d| location for info
            min_idx = np.argmin(np.abs(d))
            boundary[n] = S_grid[min_idx]
    return boundary

V_bound = []
J_bound = []
for i in range(2):
    V_bound.append(find_boundary(S_grid, V[i,:,:], futures_grid[i,:,:] - c))
    J_bound.append(find_boundary(S_grid, J[i,:,:], np.maximum(V[i,:,:] - (futures_grid[i,:,:] + c), 0)))

# --- Plot optimal boundaries ---
plt.figure(figsize=(10,8))
for i in [0,1]:
    plt.plot(t_grid * 252, V_bound[i], label=f'Liquidate (V) regime {i+1}', color=['blue','red'][i])
    plt.plot(t_grid * 252, J_bound[i], '--', label=f'Entry (J) regime {i+1}', color=['blue','red'][i])
plt.xlabel('Time (days)')
plt.ylabel('VIX')
plt.title('Optimal Long-Short Boundaries (Your Estimated Regimes)')
plt.ylim([0, Smax])
plt.grid()
plt.legend()
plt.show()