import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

# -- CIR parameters matching the paper --
mu = 0.2
theta = 18.16    # or use 40.36 for high regime
sigma = 5.33     # or use 6.42 for high regime
cost_entry = 0.5
cost_exit = 0.5
r = 0.05

# -- Strategy/FD grid setup --
T = 22                  # days to expiry (match paper)
N = T*4                 # finer grid for smooth boundary
dt = T / N
Smin, Smax = 10, 80
M = 200
ds = (Smax - Smin) / M
time_grid = np.linspace(0, T, N+1)
vix_grid = np.linspace(Smin, Smax, M+1)
V = np.zeros((M+1, N+1))   # Exit value function
J = np.zeros((M+1, N+1))   # Entry value function

# -- Terminal Condition --
for m in range(M+1):
    V[m, N] = vix_grid[m] - cost_exit
    J[m, N] = 0            # No value to entering at expiry

# -- Value PDE solve (finite difference) --
def get_coeffs(mu, th, s, sig, dt, ds, r):
    phi = mu * (th - s)
    alpha = dt/(4*ds) * (sig**2/ds - phi)
    beta = -dt/2 * (r + sig**2 / ds**2)
    gamma = dt/(4*ds) * (sig**2/ds + phi)
    return alpha, beta, gamma

for n in reversed(range(1, N+1)):
    immediate_exit = vix_grid - cost_exit
    alpha, beta, gamma = get_coeffs(mu, theta, vix_grid, sigma, dt, ds, r)
    lower = -alpha[1:]
    main = 1 - beta
    upper = -gamma[:-1]
    ab = np.zeros((3, M+1))
    ab[0,1:] = upper
    ab[1,:] = main
    ab[2,:-1] = lower

    # Exit value (solve for V[:, n-1])
    rhsV = V[:, n]
    Vsol = solve_banded((1,1), ab, rhsV)
    V[:, n-1] = np.maximum(Vsol, immediate_exit)

    # Entry value (solve for J[:, n-1])
    immediate_entry = np.maximum(V[:, n-1] - (vix_grid + cost_entry), 0)
    rhsJ = J[:, n]
    Jsol = solve_banded((1,1), ab, rhsJ)
    J[:, n-1] = np.maximum(Jsol, immediate_entry)

# -- Boundary extraction via interpolation --
def find_boundary(Vvec, vix_grid, payoff):
    """Find the VIX where value==reward (first crossing) by linear interpolation."""
    diff = Vvec - payoff
    sign_change = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_change) == 0:
        return np.nan
    idx = sign_change[0]
    x0, x1 = vix_grid[idx], vix_grid[idx+1]
    y0, y1 = diff[idx], diff[idx+1]
    if y1 == y0:
        return x0
    return x0 - y0 * (x1 - x0) / (y1 - y0)

buy_curve = np.zeros(N+1)
sell_curve = np.zeros(N+1)
for n in range(N+1):
    # BUY: where entry value agent is indifferent to wait/buy (J==V-(S+cost))
    buy_curve[n] = find_boundary(J[:, n], vix_grid, V[:, n] - vix_grid - cost_entry)
    # SELL: where exit value equals immediate reward (V == S-cost)
    sell_curve[n] = find_boundary(V[:, n], vix_grid, vix_grid - cost_exit)

# -- Simulate a CIR path --
np.random.seed(0)
def simulate_CIR(S0, mu, theta, sigma, T, N):
    dt = T/N
    S = [S0]
    for _ in range(N):
        x = S[-1]
        dW = np.random.randn() * np.sqrt(dt)
        x_new = np.abs(x + mu*(theta-x)*dt + sigma*np.sqrt(np.maximum(x,0))*dW)
        S.append(x_new)
    return np.array(S)

S0 = 15  # Try various start points!
sim_S = simulate_CIR(S0, mu, theta, sigma, T, N)

# -- Find entry/exit times (optimal stopping) --
entry_time, exit_time = None, None
for n in range(N+1):
    if entry_time is None and sim_S[n] <= buy_curve[n]:  # path crosses below BUY boundary
        entry_time = n
    if entry_time is not None and sim_S[n] >= sell_curve[n]:  # cross above SELL boundary after entry
        exit_time = n
        break

# -- Plot everything --
plt.figure(figsize=(9,6))

plt.plot(time_grid, buy_curve, 'r-', lw=2, label='Buy boundary (entry, $\\nu$)')
plt.plot(time_grid, sell_curve, 'b-', lw=2, label='Sell boundary (exit, $\\tau$)')
plt.plot(time_grid, sim_S, 'k-', lw=1.5, label='Simulated VIX path')
if entry_time is not None:
    plt.scatter(time_grid[entry_time], sim_S[entry_time], color='r', zorder=20, s=40, label='Entry ($\\nu$)')
if exit_time is not None:
    plt.scatter(time_grid[exit_time], sim_S[exit_time], color='b', zorder=20, s=40, label='Exit ($\\tau$)')
plt.xlabel("Time (days to maturity)")
plt.ylabel("VIX")
plt.legend()
plt.title("Optimal Boundaries and Sample VIX Path (Single Regime, Paper Style)")
plt.gca().invert_xaxis()
plt.xlim(T, 0)
plt.ylim(Smin, Smax)
plt.show()