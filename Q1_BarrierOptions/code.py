import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

S0 = 100.0     
K = 100.0       
B = 120.0      
T = 1.0         
r = 0.05       
sigma = 0.20    
def delta_pm(z, tau, r, sigma, sign):
    return (np.log(z) + (r + sign * 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))

def up_and_out_analytical(S, K, B, T, r, sigma):
    if S >= B:
        return 0.0 
    
    d1_k = delta_pm(S/K, T, r, sigma, 1)
    d1_b = delta_pm(S/B, T, r, sigma, 1)
    d2_k = delta_pm(S/K, T, r, sigma, -1)
    d2_b = delta_pm(S/B, T, r, sigma, -1)
    
    y1_k = delta_pm(B**2 / (K*S), T, r, sigma, 1)
    y1_b = delta_pm(B/S, T, r, sigma, 1)
    y2_k = delta_pm(B**2 / (K*S), T, r, sigma, -1)
    y2_b = delta_pm(B/S, T, r, sigma, -1)
    
    term1 = S * (si.norm.cdf(d1_k) - si.norm.cdf(d1_b))
    term2 = -S * (B/S)**(1 + 2*r/sigma**2) * (si.norm.cdf(y1_k) - si.norm.cdf(y1_b))
    term3 = -np.exp(-r*T) * K * (si.norm.cdf(d2_k) - si.norm.cdf(d2_b))
    term4 = (S/B)**(1 - 2*r/sigma**2) * np.exp(-r*T) * K * (si.norm.cdf(y2_k) - si.norm.cdf(y2_b))
    
    return term1 + term2 + term3 + term4

analytical_price = up_and_out_analytical(S0, K, B, T, r, sigma)
print(f"Analytical Price (Continuous): {analytical_price:.4f}")

M = 100000  
m = 252     
dt_mc = T / m
beta_1 = 0.5826

B_adj = B * np.exp(-beta_1 * sigma * np.sqrt(dt_mc)) 

np.random.seed(42)
Z = np.random.standard_normal((M, m))
S_paths = np.zeros((M, m + 1))
S_paths[:, 0] = S0

for i in range(1, m + 1):
    S_paths[:, i] = S_paths[:, i-1] * np.exp((r - 0.5 * sigma**2) * dt_mc + sigma * np.sqrt(dt_mc) * Z[:, i-1])

max_prices = np.max(S_paths, axis=1)
not_knocked_out = (max_prices < B_adj)

payoffs = np.maximum(S_paths[:, -1] - K, 0) * not_knocked_out
mc_adjusted_price = np.exp(-r * T) * np.mean(payoffs)
print(f"Adjusted MC Price (m={m}): {mc_adjusted_price:.4f}")

J = 120           
time_steps = 2000
dS = B / J       
dt = T / time_steps

S_space = np.linspace(0, B, J + 1)
t_space = np.linspace(0, T, time_steps + 1)

V_grid = np.zeros((time_steps + 1, J + 1))
V_grid[-1, :] = np.maximum(S_space - K, 0)
V_grid[-1, -1] = 0.0 

A = np.zeros((J - 1, J - 1))
for j in range(1, J):
    alpha = 0.5 * dt * (sigma**2 * j**2 - r * j)
    beta  = dt * (sigma**2 * j**2 + r)
    gamma = 0.5 * dt * (sigma**2 * j**2 + r * j)
    
    if j - 2 >= 0: A[j - 1, j - 2] = -alpha
    A[j - 1, j - 1] = 1 + beta
    if j < J - 1: A[j - 1, j] = -gamma

for n in range(time_steps - 1, -1, -1):
    V_grid[n, 0] = 0.0  
    V_grid[n, -1] = 0.0 
    
    rhs = V_grid[n + 1, 1:J].copy()
    V_grid[n, 1:J] = np.linalg.solve(A, rhs)

idx_S0 = int(S0 / dS)
pde_price = V_grid[0, idx_S0]
print(f"PDE Implicit Price: {pde_price:.4f}")


delta_pde = (V_grid[0, idx_S0 + 1] - V_grid[0, idx_S0 - 1]) / (2 * dS)
print(f"Delta (at S0={S0}): {delta_pde:.4f}")

S_mesh, T_mesh = np.meshgrid(S_space, t_space)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(S_mesh, T_mesh, V_grid, cmap='magma', edgecolor='none')

ax.set_title("Up and Out Barrier Option Surface $c(S, t)$")
ax.set_xlabel("Underlying Price ($S$)")
ax.set_ylabel("Time ($t$)")
ax.set_zlabel("Option Value ($V$)")
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
