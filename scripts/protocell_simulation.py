#!/usr/bin/env python3
# ============================================================
# Protocell Simulation (Thermo-RL + Autocatalysis + Lineage)
# Phase IV: reproduction, mutation, selection, and lineage tracking
# Outputs:
#   results/repro_visible_with_lineage_summary.csv
#   results/lineage_table.csv
# ============================================================

import numpy as np
import pandas as pd
import os

# ------------------------------
# 1) Parameters
# ------------------------------
T = 50000
N = 24
M = 48
R0 = 20.0
supply0, supply_amp, supply_period = 2.0, 1.5, 600
env_noise = 0.3
patch_diffusion = 0.2
resource_decay = 0.02

# Agent energetics & state
cell_decay = 0.12
maint_cost = 0.02
move_cost = 0.015
x_divide = 1.0
death_threshold_x = 0.03
death_threshold_e = 0.02

# Policy & learning
feat_dim = 6                               # [S_int, X, E, S_local, dS_local, P]
eta = 0.008
init_theta_scale = 0.12
theta_clip = 3.0

# Thermodynamics
kT = 1.0
mu0_env = 0.0
mu0_int = 0.0
eps_floor = 1e-12
lambda_ctrl = 0.02
lambda_diss = 0.02

# Cooperation
c_share = 0.02
lambda_P = 0.1
gamma = 0.12

# Chemistry (reversible mass-action): forward-favored R2 (under this convention)
kf1 = 0.12; dG1 = -3.0
kf2 = 0.25; dG2 = +1.0
kr1 = kf1 * np.exp(-dG1/kT)
kr2 = kf2 * np.exp(-dG2/kT)

# Environmental energy
E_env0 = 0.14
E_env_amp = 0.08

# Evolution
mut_scale = 0.06

# Logging
log_every = 10
rng = np.random.default_rng(77)

# Output dir
os.makedirs("results", exist_ok=True)

# ------------------------------
# 2) Helper functions
# ------------------------------
def sigmoid(z): return 1.0/(1.0+np.exp(-z))
def tanh(z): return np.tanh(z)
def seasonal_supply(t): return supply0 + supply_amp*np.sin(2*np.pi*t/supply_period)
def seasonal_energy(t): return E_env0 + E_env_amp*np.sin(2*np.pi*t/supply_period + np.pi/3)
def chem_potential(c, mu0=0.0):
    c = np.maximum(c, eps_floor)
    return mu0 + np.log(c)

# ------------------------------
# 3) Initialization
# ------------------------------
S_env = np.full(M, R0 / M)
prev_S_env = S_env.copy()
pos = rng.integers(0, M, size=N)
S_int = rng.uniform(0.1, 0.6, N)
X = rng.uniform(0.2, 0.8, N)
E = rng.uniform(0.2, 0.8, N)
W = rng.uniform(0.0, 0.2, N)

theta_u = rng.normal(0, init_theta_scale, (N, feat_dim))
theta_m = rng.normal(0, init_theta_scale, (N, feat_dim))
P = 0.0

# Lineage tracking
agent_id = np.arange(N, dtype=int)      # unique IDs
parent_id = np.full(N, -1, dtype=int)   # -1 = no parent (roots)
generation = np.zeros(N, dtype=int)
root_id = agent_id.copy()
next_id = N
lineage_log = []

# Time series log
log = []
births = deaths = 0

# ------------------------------
# 4) Simulation
# ------------------------------
for t in range(T):
    # Environment dynamics
    S_env += seasonal_supply(t)/M + rng.normal(0, env_noise, M)
    S_env = np.maximum(S_env, 0.0)
    S_env = (1 - patch_diffusion)*S_env + 0.5*patch_diffusion*(np.roll(S_env,1) + np.roll(S_env,-1))
    S_env *= (1 - resource_decay)

    # Observations & features
    S_local = S_env[pos]
    dS_local = S_local - prev_S_env[pos]
    prev_S_env = S_env.copy()
    feats = np.stack([S_int, X, E, S_local, dS_local, np.full(N, P)], axis=1)

    # Policies
    z_u = np.sum(theta_u * feats, axis=1)
    z_m = np.sum(theta_m * feats, axis=1)
    a_u = sigmoid(z_u); a_m = tanh(z_m)

    # Movement
    move = np.zeros(N, dtype=int)
    move[a_m > 0.33] = 1
    move[a_m < -0.33] = -1
    pos = (pos + move) % M

    # Public pool
    P = (1 - lambda_P)*P + np.sum(c_share * X)
    eff = 1.0 + gamma * P

    # Thermodynamic uptake (S_env -> S_int), conserved per patch
    mu_env = chem_potential(S_env[pos], mu0_env)
    mu_int = chem_potential(S_int, mu0_int)
    delta_mu = mu_env - mu_int
    permeability = 0.1 + 0.9*a_u
    demand = np.maximum(permeability * delta_mu, 0.0) * eff

    demand_per_patch = np.zeros(M)
    for i in range(N):
        demand_per_patch[pos[i]] += demand[i]

    inflow_S = np.zeros(N)
    for i in range(N):
        d = demand_per_patch[pos[i]]
        avail = S_env[pos[i]]
        take = 0.0 if d <= 1e-12 or avail <= 1e-12 else (demand[i]/d) * min(avail, d)
        inflow_S[i] = take
        S_env[pos[i]] -= take

    # Energy input
    E += seasonal_energy(t) + rng.normal(0, 0.005, N)
    E = np.maximum(E, 0.0)

    # Reversible reactions
    prod1 = X*W + 1e-12
    reac1 = S_int*E + 1e-12
    J1 = kf1*reac1 - kr1*prod1
    A1 = np.log(reac1/(prod1)) + dG1/kT

    prod2 = (X*X*W) + 1e-12
    reac2 = (X*E) + 1e-12
    J2 = kf2*reac2 - kr2*prod2
    A2 = np.log(reac2/(prod2)) + dG2/kT

    sigma = float(np.sum(J1*A1 + J2*A2))

    # State updates
    S_int = np.maximum(S_int + inflow_S - J1, 0.0)
    X = np.maximum(X + J1 + J2 - cell_decay*X, 0.0)
    E = np.maximum(E - J1 - J2 - maint_cost - move_cost*np.abs(move), 0.0)
    W = np.maximum(W + J1 + J2 - 0.05*W, 0.0)

    # Reward & policy updates
    r_viab = - (X - 1.0)**2
    r_ctrl = - lambda_ctrl * (a_u**2 + a_m**2)
    r = r_viab + r_ctrl - (lambda_diss * sigma)/N

    theta_u += eta * r[:,None] * (a_u*(1-a_u))[:,None] * feats
    theta_m += eta * r[:,None] * (1 - a_m**2)[:,None] * feats
    theta_u = np.clip(theta_u, -theta_clip, theta_clip)
    theta_m = np.clip(theta_m, -theta_clip, theta_clip)

    # Reproduction & death
    for i in range(N):
        if X[i] < death_threshold_x or E[i] < death_threshold_e:
            deaths += 1
            same = np.where(pos == pos[i])[0]
            j = same[np.argmax(X[same])] if len(same) > 1 else np.argmax(X)
            S_int[i] = 0.5 * max(S_int[j], 0.1)
            X[i] = 0.6 * max(X[j], 0.1)
            E[i] = 0.6 * max(E[j], 0.1)
            W[i] = 0.2 * W[j]
            theta_u[i] = theta_u[j] + rng.normal(0, mut_scale, feat_dim)
            theta_m[i] = theta_m[j] + rng.normal(0, mut_scale, feat_dim)
            # keep lineage labels of the reseeded ancestor
            root_id[i] = root_id[j]
            parent_id[i] = parent_id[j]
            generation[i] = generation[j]
            agent_id[i] = agent_id[i]  # keep same id on reseed (not a birth)
        elif X[i] > x_divide and E[i] > 0.35:
            births += 1
            j = np.argmin(X)  # replace weakest slot with child
            parent = i
            child = j

            # split states
            S_int_child = 0.6 * S_int[parent]
            X_child = 0.6 * X[parent]
            E_child = 0.6 * E[parent]
            W_child = 0.5 * W[parent]
            S_int[parent] *= 0.6; X[parent] *= 0.6; E[parent] *= 0.6; W[parent] *= 0.5

            theta_u_child = theta_u[parent] + rng.normal(0, mut_scale, feat_dim)
            theta_m_child = theta_m[parent] + rng.normal(0, mut_scale, feat_dim)

            # assign to child slot
            S_int[child], X[child], E[child], W[child] = S_int_child, X_child, E_child, W_child
            theta_u[child], theta_m[child] = theta_u_child, theta_m_child
            pos[child] = (pos[parent] + rng.integers(-1,2)) % M

            # lineage bookkeeping
            child_new_id = next_id; next_id += 1
            lineage_log.append({
                "time": t,
                "child_id": int(child_new_id),
                "parent_id": int(agent_id[parent]),
                "generation": int(generation[parent] + 1),
                "patch": int(pos[parent]),
                "parent_X": float(X[parent]),
                "parent_E": float(E[parent]),
                "mean_reward_pop": float(np.mean(r))
            })
            parent_id[child] = agent_id[parent]
            agent_id[child] = child_new_id
            generation[child] = generation[parent] + 1
            root_id[child] = root_id[parent]

    # Time-series log
    if t % log_every == 0:
        log.append({
            "time": t,
            "resource_total": float(np.sum(S_env)),
            "mean_Sint": float(np.mean(S_int)),
            "mean_X": float(np.mean(X)),
            "mean_E": float(np.mean(E)),
            "public_pool": float(P),
            "theta_u_var": float(np.var(theta_u)),
            "theta_m_var": float(np.var(theta_m)),
            "entropy_prod": float(sigma),
            "births": births,
            "deaths": deaths
        })

# ------------------------------
# 5) Save outputs
# ------------------------------
pd.DataFrame(log).to_csv("results/repro_visible_with_lineage_summary.csv", index=False)
pd.DataFrame(lineage_log).to_csv("results/lineage_table.csv", index=False)

theta_summary = pd.DataFrame({
    "agent_id": np.arange(N),
    "mean_theta_u": np.mean(theta_u, axis=1),
    "mean_theta_m": np.mean(theta_m, axis=1),
    "root_id": root_id,
    "generation": generation
})
theta_summary.to_csv("results/final_theta.csv", index=False)

print("Simulation complete.")
print("Saved: results/repro_visible_with_lineage_summary.csv")
print("Saved: results/lineage_table.csv")
