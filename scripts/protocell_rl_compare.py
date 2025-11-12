#!/usr/bin/env python3
# ============================================================
# Protocell Evolution Simulator (Thermo-RL, A2C, REINFORCE, Random)
# Publication-quality version with multi-seed averaging, safe math,
# full warning/error logging, and single multi-panel figure.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys, datetime, warnings

# ============================================================
# 1. Configuration
# ============================================================
T = 5000
N = 24
M = 48
R0 = 20.0
supply0, supply_amp, supply_period = 2.0, 1.5, 600
env_noise = 0.3
patch_diffusion = 0.2
resource_decay = 0.02

cell_decay = 0.12
maint_cost = 0.02
move_cost = 0.015
x_divide = 1.0
death_threshold_x = 0.03
death_threshold_e = 0.02

feat_dim = 6
eta = 0.008
init_theta_scale = 0.12
theta_clip = 3.0

kT = 1.0
mu0_env = 0.0
mu0_int = 0.0
eps_floor = 1e-12
lambda_ctrl = 0.02
lambda_diss = 0.02

c_share = 0.02
lambda_P = 0.1
gamma = 0.12

kf1 = 0.12; dG1 = -3.0
kf2 = 0.25; dG2 = +1.0
kr1 = kf1 * np.exp(-dG1/kT)
kr2 = kf2 * np.exp(-dG2/kT)

E_env0 = 0.14
E_env_amp = 0.08

mut_scale = 0.06
log_every = 10

a2c_gamma = 0.9
a2c_v_lr = 0.1

SEEDS = [42, 77, 123, 314, 999]
MODES = ["thermo_rl", "reinforce", "a2c", "random"]

os.makedirs("results", exist_ok=True)

# ============================================================
# 2. Logging setup (stdout, stderr, warnings)
# ============================================================
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"results/simulation_log_{timestamp}.txt"

log_fh = open(log_file, "w", buffering=1)
sys.stdout = log_fh
sys.stderr = log_fh

warnings.simplefilter("default")
warnings.showwarning = lambda msg, cat, fname, lineno, file=None, line=None: \
    print(f"[WARNING] {cat.__name__}: {msg} (line {lineno} in {fname})", file=sys.stderr)

print(f"=== Protocell Simulation Log ({timestamp}) ===")
print(f"All stdout, stderr, and warnings redirected to {log_file}\n")

# ============================================================
# 3. Numeric safety helpers
# ============================================================
def safe_exp(x, clip=50.0):
    return np.exp(np.clip(x, -clip, clip))

def safe_log(x, floor=1e-12):
    return np.log(np.maximum(x, floor))

def safe_sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))

# ============================================================
# 4. Helper functions
# ============================================================
def tanh(z): return np.tanh(z)
def seasonal_supply(t): return supply0 + supply_amp * np.sin(2*np.pi*t/supply_period)
def seasonal_energy(t): return E_env0 + E_env_amp * np.sin(2*np.pi*t/supply_period + np.pi/3)
def chem_potential(c, mu0=0.0): return mu0 + safe_log(c)

# ============================================================
# 5. Core simulation loop
# ============================================================
def run_simulation(rl_mode, seed):
    rng = np.random.default_rng(seed)

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
    V = np.zeros(N)

    births = deaths = 0
    log = []

    for t in range(T):
        S_env += seasonal_supply(t)/M + rng.normal(0, env_noise, M)
        S_env = np.maximum(S_env, 0.0)
        S_env = (1 - patch_diffusion)*S_env + 0.5*patch_diffusion*(np.roll(S_env,1) + np.roll(S_env,-1))
        S_env *= (1 - resource_decay)

        S_local = S_env[pos]
        dS_local = S_local - prev_S_env[pos]
        prev_S_env = S_env.copy()
        feats = np.stack([S_int, X, E, S_local, dS_local, np.full(N, P)], axis=1)

        z_u = np.sum(theta_u * feats, axis=1)
        z_m = np.sum(theta_m * feats, axis=1)
        a_u = safe_sigmoid(z_u)
        a_m = np.tanh(np.clip(z_m, -5, 5))

        move = np.zeros(N, dtype=int)
        move[a_m > 0.33] = 1
        move[a_m < -0.33] = -1
        pos = (pos + move) % M

        P = (1 - lambda_P)*P + np.sum(c_share * X)
        eff = 1.0 + gamma * P

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
            if d > 1e-12 and avail > 1e-12:
                take = (demand[i]/d) * min(avail, d)
            else:
                take = 0.0
            inflow_S[i] = take
            S_env[pos[i]] -= take

        E += seasonal_energy(t) + rng.normal(0, 0.005, N)
        E = np.maximum(E, 0.0)

        prod1 = X*W + 1e-12
        reac1 = S_int*E + 1e-12
        J1 = kf1*reac1 - kr1*prod1
        A1 = safe_log(reac1/(prod1)) + dG1/kT

        prod2 = (X*X*W) + 1e-12
        reac2 = (X*E) + 1e-12
        J2 = kf2*reac2 - kr2*prod2
        A2 = safe_log(reac2/(prod2)) + dG2/kT

        sigma = float(np.sum(J1*A1 + J2*A2))

        # Update and clip
        S_int = np.maximum(S_int + inflow_S - J1, 0.0)
        X = np.maximum(X + J1 + J2 - cell_decay*X, 0.0)
        E = np.maximum(E - J1 - J2 - maint_cost - move_cost*np.abs(move), 0.0)
        W = np.maximum(W + J1 + J2 - 0.05*W, 0.0)
        S_int = np.clip(S_int, 0, 10)
        X = np.clip(X, 0, 10)
        E = np.clip(E, 0, 10)
        W = np.clip(W, 0, 10)

        # Reward
        r_viab = - np.square(X - 1.0)
        r_ctrl = - lambda_ctrl * (a_u**2 + a_m**2)
        r = np.clip(r_viab + r_ctrl, -10, 10)
        if rl_mode == "thermo_rl":
            r -= (lambda_diss * sigma)/N

        if rl_mode == "thermo_rl":
            theta_u += eta * r[:,None] * (a_u*(1-a_u))[:,None] * feats
            theta_m += eta * r[:,None] * (1 - a_m**2)[:,None] * feats
        elif rl_mode == "reinforce":
            adv = r - np.mean(r)
            theta_u += eta * adv[:,None] * (a_u*(1-a_u))[:,None] * feats
            theta_m += eta * adv[:,None] * (1 - a_m**2)[:,None] * feats
        elif rl_mode == "a2c":
            td_target = r + a2c_gamma * V
            td_error = np.clip(td_target - V, -10, 10)
            V += a2c_v_lr * td_error
            theta_u += eta * td_error[:,None] * (a_u*(1-a_u))[:,None] * feats
            theta_m += eta * td_error[:,None] * (1 - a_m**2)[:,None] * feats

        theta_u = np.clip(theta_u, -theta_clip, theta_clip)
        theta_m = np.clip(theta_m, -theta_clip, theta_clip)

        for i in range(N):
            if X[i] < death_threshold_x or E[i] < death_threshold_e:
                deaths += 1
                j = np.argmax(X)
                S_int[i] = 0.5 * S_int[j]
                X[i] = 0.6 * X[j]
                E[i] = 0.6 * E[j]
                W[i] = 0.2 * W[j]
            elif X[i] > x_divide and E[i] > 0.35:
                births += 1
                j = np.argmin(X)
                S_int[j] = 0.6 * S_int[i]
                X[j] = 0.6 * X[i]
                E[j] = 0.6 * E[i]
                W[j] = 0.5 * W[i]

        if t % log_every == 0:
            log.append({
                "time": t,
                "mean_reward": np.nanmean(r),
                "entropy_prod": np.nan_to_num(sigma),
                "mean_X": np.nanmean(X),
                "mean_E": np.nanmean(E),
                "births": births,
                "deaths": deaths,
                "rl_mode": rl_mode,
                "seed": seed
            })

    return pd.DataFrame(log)

# ============================================================
# 6. Multi-seed execution and plotting
# ============================================================
def run_all():
    all_runs = []
    for mode in MODES:
        for seed in SEEDS:
            print(f"▶ Running {mode.upper()} (seed={seed})")
            df = run_simulation(mode, seed)
            all_runs.append(df)
    df_all = pd.concat(all_runs, ignore_index=True)
    df_all.to_csv("results/rl_algorithm_comparison.csv", index=False)
    return df_all

def plot_combined(df_all):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.frameon": True,
        "lines.linewidth": 2.0
    })

    metrics = ["mean_X", "mean_reward", "entropy_prod", "births", "deaths"]
    titles = ["Biomass (X)", "Reward", "Entropy Production", "Births", "Deaths"]
    colors = {
        "thermo_rl": "#d62728",
        "reinforce": "#1f77b4",
        "a2c": "#2ca02c",
        "random": "#7f7f7f"
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4), sharex=True)
    for ax, metric, title in zip(axes, metrics, titles):
        for mode in MODES:
            sub = df_all[df_all["rl_mode"] == mode]
            grouped = sub.groupby("time")[metric]
            mean = grouped.mean()
            sem = grouped.sem()
            ax.plot(mean.index, mean.values, label=mode.upper(), color=colors[mode])
            ax.fill_between(mean.index, mean - sem, mean + sem, color=colors[mode], alpha=0.2)
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value" if metric not in ["births", "deaths"] else "Count")
    axes[0].legend(loc="best")
    plt.tight_layout()
    fig.savefig("results/protocell_rl_comparison_pub.png", dpi=300)
    fig.savefig("results/protocell_rl_comparison_pub.pdf")
    print("\n📊 Saved combined multi-panel figure to results/protocell_rl_comparison_pub.[png,pdf]")

# ============================================================
# 7. Main
# ============================================================
if __name__ == "__main__":
    df_all = run_all()
    plot_combined(df_all)
    print("\n✅ All simulations complete.")
    print(f"Log written to {log_file}")
