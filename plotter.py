import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  # for reproducibility

# === 1. PARAMETERS ===
# Time parameters (in years)
T_hat = 22/252   # Exercise window (years)
N_t = 80         # time steps
N_s = 120        # space (VIX) steps

Smin, Smax = 0.0, 55.0
dt = T_hat / N_t
ds = (Smax - Smin) / N_s
grid_s = np.linspace(Smin, Smax, N_s+1)
grid_t = np.linspace(0, T_hat, N_t+1)
regimes = [0, 1]

# CIR parameters (from paper) for regimes 1 and 2
mu     = [8.57, 9.0]
theta  = [17.58, 39.5]
sigma  = [5.33, 6.42]
mu_tilde = [4.55, 4.59]
theta_tilde = [18.16, 40.36]
sigma_tilde = [5.33, 6.42]

r = 0.05  # Discount rate

# Generator (Q matrix) for Markov chain
q12, q21 = 0.1, 0.5
Q = np.array([[-q12, q12], [q21, -q21]])

# Transaction costs
c_exit = 0.01  # at liquidation

# =========== 2. FUTURES PRICE CRANK–NICOLSON SOLVER ==============
def solve_futures_price_CN(mu, theta, sigma, Q, T, N_t, N_s, Smin, Smax):
    dt = T / N_t
    ds = (Smax - Smin) / N_s
    S = np.linspace(Smin, Smax, N_s+1)
    f = np.zeros((2, N_t+1, N_s+1))  # (regime, time, space)

    # terminal: f(T, s, i) = s
    for i in [0, 1]:
        f[i, -1, :] = S

    for n in reversed(range(N_t)):
        for i in [0, 1]:
            vals = f[i, n+1, :]
            main = np.ones(N_s-1)
            upper = np.zeros(N_s-2)
            lower = np.zeros(N_s-2)
            # Coefficients
            g = S[1:-1]
            a = 0.25*dt*(sigma[i]**2 * g/ds**2 - mu[i]*(theta[i] - g)/ds)
            b = -0.5*dt*(sigma[i]**2 * g/ds**2) - dt*0.5*(Q[i,i])
            c = 0.25*dt*(sigma[i]**2 * g/ds**2 + mu[i]*(theta[i] - g)/ds)
            main += b
            upper[:] = a[1:]
            lower[:] = c[:-1]
            M1 = np.diag(main) + np.diag(upper,1) + np.diag(lower,-1)

            # rhs: CN contributions, regime coupling
            rhs = (vals[1:-1]
                + 0.25*dt*(sigma[i]**2 * g * (vals[2:] - 2*vals[1:-1] + vals[:-2])/ds**2
                           + mu[i]*(theta[i] - g)*(vals[2:] - vals[:-2])/(2*ds))
                + 0.5*dt*(Q[i,1-i])*f[1-i, n+1, 1:-1])

            rhs[0]  -= c[0]*S[0]
            rhs[-1] -= a[-1]*S[-1]

            sol = np.linalg.solve(M1, rhs)
            f[i, n, 1:-1] = sol
            f[i, n, 0] = S[0]
            f[i, n, -1] = S[-1]
    return f

# =========== 3. PSOR SOLVER FOR VARIATIONAL INEQUALITY ==============

def solve_vi_psor_CN(mu, theta, sigma, Q, r, 
                     f_exo, reward, exercise_mask, 
                     T_hat, N_t, N_s, Smin, Smax, 
                     omega=1.2, tol=1e-6, max_iter=300):
    dt = T_hat / N_t
    ds = (Smax - Smin) / N_s
    S = np.linspace(Smin, Smax, N_s+1)
    v = np.zeros((2, N_t+1, N_s+1))
    for i in [0, 1]:
        v[i, -1, :] = reward[i, -1, :]
    for n in reversed(range(N_t)):
        for i in [0, 1]:
            vals = v[i, n+1, :]
            g = S[1:-1]
            a = 0.25*dt*(sigma[i]**2 * g / ds**2 - mu[i]*(theta[i] - g)/ds)
            b_diag = 1 + 0.5*dt*(r + sigma[i]**2 * g / ds**2 - Q[i,i])
            c = 0.25*dt*(sigma[i]**2 * g / ds**2 + mu[i]*(theta[i] - g)/ds)
            M1 = np.diag(b_diag) + np.diag(-a[1:],1) + np.diag(-c[:-1],-1)
            rhs = (vals[1:-1]
                 + 0.25*dt*(sigma[i]**2 * g * (vals[2:] - 2*vals[1:-1] + vals[:-2]) / ds**2
                            + mu[i]*(theta[i] - g)*(vals[2:] - vals[:-2])/(2*ds))
                 + 0.5*dt*Q[i,1-i]*v[1-i, n+1, 1:-1])
            rhs[0]  -= c[0]*S[0]     # boundary
            rhs[-1] -= a[-1]*S[-1]   # boundary
            x = vals[1:-1].copy()
            for it in range(max_iter):
                error = 0
                for j in range(N_s-1):
                    sum1 = 0
                    if j>0: sum1 -= c[j-1]*x[j-1]
                    if j<N_s-2: sum1 -= a[j]*x[j+1]
                    num = (rhs[j] + sum1)
                    den = b_diag[j]
                    x_new = (1-omega)*x[j] + omega*num/den
                    # Project
                    if exercise_mask[i, n, j+1]:
                        x_new = reward[i, n, j+1]
                    else:
                        x_new = max(x_new, reward[i, n, j+1])
                    error = max(error, abs(x[j] - x_new))
                    x[j] = x_new
                if error < tol:
                    break
            v[i, n, 1:-1] = x
            v[i, n, 0] = reward[i, n, 0]
            v[i, n, -1] = reward[i, n, -1]
    return v


# =========== 4. BOUNDARY EXTRACTION ==============
def extract_boundary(v, reward, grid_s):
    reg_count = v.shape[0]
    N_t = v.shape[1] - 1
    boundary_s = np.zeros((reg_count, N_t+1))
    for i in range(reg_count):
        for n in range(N_t+1):
            diff = v[i, n] - reward[i, n]
            idxs = np.where(diff <= 1e-5)[0]
            if len(idxs) > 0:
                boundary_s[i, n] = grid_s[idxs[0]]
            else:
                boundary_s[i, n] = grid_s[-1]
    return boundary_s


# =========== 5. SIMULATION + PLOTTING ==============
# 1. Solve for futures price (risk-neutral measure)
f = solve_futures_price_CN(mu_tilde, theta_tilde, sigma_tilde, Q, T_hat, N_t, N_s, Smin, Smax)

# 2. Define liquidation reward function (minus cost)
reward = np.zeros_like(f)
for i in [0, 1]:
    reward[i,:,:] = f[i,:,:] - c_exit

    

# 3. Exercise mask ("False" everywhere for normal stopping problem)
exercise_mask = np.zeros_like(reward, dtype=bool)

# 4. Solve value function with PSOR
v = solve_vi_psor_CN(mu, theta, sigma, Q, r, f, reward, exercise_mask,
                     T_hat, N_t, N_s, Smin, Smax, omega=1.2)

# 5. Extract stopping (exit) boundaries
boundary_s = extract_boundary(v, reward, grid_s)

# 6. Plot boundaries
plt.figure(figsize=(9,6))
plt.plot(grid_t*252, boundary_s[0,:], 'b', lw=2, label='Boundary Regime 1')
plt.plot(grid_t*252, boundary_s[1,:], 'r', lw=2, label='Boundary Regime 2')
plt.xlabel('Time to Expiry (days)')
plt.ylabel('VIX Level')
plt.legend()
plt.title('Optimal Exit Boundaries for 2-State Regime Switching CIR')
plt.ylim(Smin, Smax)
plt.tight_layout()
plt.show()


# ========== 6. Regime-Switching CIR PATH SIMULATION & OVERLAY ===========
def simulate_path(T_hat, N_t, S0, mu, theta, sigma, Q):
    dt = T_hat / N_t
    t, S, regimes = np.zeros(N_t+1), np.zeros(N_t+1), np.zeros(N_t+1, dtype=int)
    t[0], S[0], regimes[0] = 0, S0, 0
    for n in range(N_t):
        reg = regimes[n]
        # Markov regime switch
        if np.random.rand() < abs(Q[reg,1-reg])*dt:
            reg = 1 - reg
        regimes[n+1] = reg
        # CIR step
        dW = np.sqrt(dt)*np.random.randn()
        S[n+1] = max(S[n] + mu[reg]*(theta[reg] - S[n])*dt + sigma[reg]*np.sqrt(max(S[n],0))*dW, 0.001)
        t[n+1] = t[n] + dt
    return t*252, S, regimes

# Simulate one path, starting at VIX = 18
t_days, S_path, regime_path = simulate_path(T_hat, N_t, 18.0, mu, theta, sigma, Q)

plt.figure(figsize=(10,6))
plt.plot(t_days, S_path, color='k', lw=1.8, label='Simulated VIX Path')
plt.plot(grid_t*252, boundary_s[0,:], '--b', lw=1.5, label='Boundary Regime 1')
plt.plot(grid_t*252, boundary_s[1,:], '--r', lw=1.5, label='Boundary Regime 2')

# Color background for regime
for k in range(len(t_days)-1):
    color = 'cyan' if regime_path[k]==0 else 'salmon'
    plt.axvspan(t_days[k], t_days[k+1], color=color, alpha=0.08, lw=0)
plt.xlabel('Time (days)')
plt.ylabel('VIX Level')
plt.title('Simulated VIX path with regime switching and optimal exit boundaries')
plt.legend()
plt.ylim(Smin, Smax)
plt.tight_layout()
plt.show()