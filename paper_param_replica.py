import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

# ------- Manual paper parameters for CIR regimes ------
regimes = 2
mu_array      = np.array([0.2, 0.2])
theta_array   = np.array([18.16, 40.36])
sigma_array   = np.array([5.33, 6.42])
# For more visual variety you could try [0.2,0.2], [17.6,39.5], [5.3,6.4]

# "Generator" Q for two-state Markov (paper uses q12=0.1, q21=0.5)
Q = np.array([[-0.1, 0.1],
              [ 0.5, -0.5]])

cost_entry = 5  # Try 0 and 0.01 to see Figure 2 left/right
cost_exit  = 5
r = 0.05

T = 66    # days to expiry
N = T*2
dt = T/N
Smin, Smax = 10, 80
M = 100
ds = (Smax - Smin)/M
time_grid = np.linspace(0, T, N+1)
vix_grid = np.linspace(Smin, Smax, M+1)

num_regimes = regimes
V = np.zeros((num_regimes, M+1, N+1)) # Exit value function
J = np.zeros((num_regimes, M+1, N+1)) # Entry value function

# --------- 5. TERMINAL CONDITION -----------
for reg in range(num_regimes):
    for m in range(M+1):
        V[reg, m, N] = vix_grid[m] - cost_exit
        J[reg, m, N] = 0

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
            elif np.abs(V[reg, m, n] - (vix_grid[m] - cost_exit)) < 1e-5:
                region_map[reg, m, n] = -1
            else:
                region_map[reg, m, n] = 0  # WAIT


buy_boundary = np.full((num_regimes, N+1), np.nan)
sell_boundary = np.full((num_regimes, N+1), np.nan)
for reg in range(num_regimes):
    for n in range(N+1):
        buy_idxs = np.where(region_map[reg, :, n] == 1)[0]
        if buy_idxs.size > 0:
            buy_boundary[reg, n] = vix_grid[buy_idxs[0]]
        sell_idxs = np.where(region_map[reg, :, n] == -1)[0]
        if sell_idxs.size > 0:
            sell_boundary[reg, n] = vix_grid[sell_idxs[-1]]

fig, axs = plt.subplots(num_regimes, 1, figsize=(11,7), sharex=True)
for reg in range(num_regimes):
    im = axs[reg].imshow(region_map[reg,:,:], aspect='auto', cmap='bwr',
                         extent=[T, 0, Smin, Smax], origin='lower',
                         vmin=-1, vmax=1)
    axs[reg].set_title(f'Regime {reg+1}: Optimal Trading/Entry Regions with Boundaries')
    axs[reg].set_ylabel('VIX')
    cbar = plt.colorbar(im, ax=axs[reg], orientation='vertical', ticks=[-1,0,1])
    cbar.ax.set_yticklabels(['SELL','WAIT','BUY'])
    axs[reg].plot(time_grid, buy_boundary[reg], 'k--', label='Buy entry boundary')
    axs[reg].plot(time_grid, sell_boundary[reg], 'g--', label='Sell exit boundary')
    axs[reg].legend()
axs[-1].set_xlabel('Time to Expiry (days)')
plt.tight_layout()
plt.show()