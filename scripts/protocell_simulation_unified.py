#!/usr/bin/env python3
# ============================================================
# Protocell Master Simulation (Thermo-RL + Autocatalysis + Lineage + InfoMetrics)
# Fully Integrated Multi-Seed, Vectorized, and Publication-Ready
# ============================================================

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import ttest_ind, kruskal
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# ============================================================
# 0️⃣ GLOBAL CONFIG & ENV OVERRIDES
# ============================================================

def get_bool(name, default=False):
    v = os.getenv(name, str(default))
    return v.lower() in ("true", "1", "yes")

def get_float(name, default):
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default

# Environment parameter overrides
disable_rl = get_bool("DISABLE_RL", False)
disable_info = get_bool("DISABLE_INFO_METRICS", False)
gamma = get_float("GAMMA", 0.12)
c_share = get_float("C_SHARE", 0.02)
lambda_diss = get_float("LAMBDA_DISS", 0.02)
env_noise = get_float("ENV_NOISE", 0.3)

# ============================================================
# 1️⃣ MODEL PARAMETERS
# ============================================================

T = 5000
N = 24
M = 48
R0 = 20.0
supply0, supply_amp, supply_period = 2.0, 1.5, 600
patch_diffusion = 0.2
resource_decay = 0.02

# Energetics
cell_decay, maint_cost, move_cost = 0.12, 0.02, 0.015
x_divide, death_threshold_x, death_threshold_e = 1.0, 0.03, 0.02

# Learning
feat_dim, eta, init_theta_scale, theta_clip = 6, 0.008, 0.12, 3.0

# Thermodynamics
kT = 1.0
mu0_env = mu0_int = 0.0
eps_floor = 1e-12
lambda_ctrl = 0.02

# Cooperation
lambda_P = 0.1

# Chemistry
kf1, dG1 = 0.12, -3.0
kf2, dG2 = 0.25, +1.0
kr1, kr2 = kf1*np.exp(-dG1/kT), kf2*np.exp(-dG2/kT)

# Environment energy
E_env0, E_env_amp = 0.14, 0.08

# Mutation
mut_scale = 0.06

# Logging
log_every = 10

# Seeds
SEEDS = [11, 22, 33, 44, 55]

os.makedirs("results", exist_ok=True)

# ============================================================
# 2️⃣ HELPER FUNCTIONS
# ============================================================

def sigmoid(z): return 1/(1+np.exp(-z))
def tanh(z): return np.tanh(z)
def seasonal_supply(t): return supply0 + supply_amp*np.sin(2*np.pi*t/supply_period)
def seasonal_energy(t): return E_env0 + E_env_amp*np.sin(2*np.pi*t/supply_period + np.pi/3)
def chem_potential(c, mu0=0.0): return mu0 + np.log(np.maximum(c, eps_floor))

def safe_entropy(x, bins=10):
    if np.allclose(x, x[0]): return 0.0
    hist, _ = np.histogram(x, bins=bins, density=True)
    p = hist / np.sum(hist)
    p = p[p>0]
    return float(-np.sum(p*np.log(p)))  # nats

def safe_mutual_info(x, y, bins=10):
    if np.allclose(x, x[0]) or np.allclose(y, y[0]): return 0.0
    hist_xy, _, _ = np.histogram2d(x, y, bins=bins, density=True)
    pxy = hist_xy / np.sum(hist_xy)
    px, py = np.sum(pxy, axis=1), np.sum(pxy, axis=0)
    pxy, px, py = pxy+1e-12, px+1e-12, py+1e-12
    return float(np.sum(pxy * np.log(pxy/(px[:,None]*py[None,:]))))

# ============================================================
# 3️⃣ PROTOCELL SIMULATION
# ============================================================

def run_protocell_simulation(seed=77):
    rng = np.random.default_rng(seed)
    S_env = np.full(M, R0/M)
    prev_S_env = S_env.copy()
    pos = rng.integers(0, M, N)
    S_int = rng.uniform(0.1, 0.6, N)
    X = rng.uniform(0.2, 0.8, N)
    E = rng.uniform(0.2, 0.8, N)
    W = rng.uniform(0.0, 0.2, N)
    theta_u = rng.normal(0, init_theta_scale, (N, feat_dim))
    theta_m = rng.normal(0, init_theta_scale, (N, feat_dim))
    P = 0.0

    agent_id = np.arange(N, dtype=int)
    parent_id = np.full(N, -1, dtype=int)
    generation = np.zeros(N, dtype=int)
    root_id = agent_id.copy()
    next_id = N

    log, lineage_log = [], []
    births = deaths = 0
    theta_prev_u = theta_u.copy()
    theta_prev_m = theta_m.copy()

    for t in range(T):
        # --- Environment ---
        S_env += seasonal_supply(t)/M + rng.normal(0, env_noise, M)
        S_env = np.maximum(S_env, 0.0)
        S_env = (1-patch_diffusion)*S_env + 0.5*patch_diffusion*(np.roll(S_env,1)+np.roll(S_env,-1))
        S_env *= (1-resource_decay)

        # --- Agent features ---
        S_local = S_env[pos]
        dS_local = S_local - prev_S_env[pos]
        prev_S_env = S_env.copy()
        feats = np.stack([S_int, X, E, S_local, dS_local, np.full(N, P)], axis=1)

        # --- Policy ---
        z_u, z_m = np.sum(theta_u*feats,1), np.sum(theta_m*feats,1)
        a_u, a_m = sigmoid(z_u), tanh(z_m)
        move = np.zeros(N, int)
        move[a_m>0.33]=1; move[a_m<-0.33]=-1
        pos = (pos+move)%M

        # --- Cooperation ---
        P = (1-lambda_P)*P + np.sum(c_share*X)
        eff = 1.0 + gamma*P

        # --- Uptake (vectorized) ---
        mu_env, mu_int = chem_potential(S_env[pos]), chem_potential(S_int)
        delta_mu = mu_env - mu_int
        permeability = 0.1 + 0.9*a_u
        demand = np.maximum(permeability*delta_mu,0)*eff

        demand_per_patch = np.zeros(M)
        np.add.at(demand_per_patch, pos, demand)
        avail = S_env[pos]
        denom = demand_per_patch[pos]
        frac = np.where((denom>1e-12)&(avail>1e-12), np.minimum(1.0, avail/denom), 0.0)
        inflow_S = demand * frac
        taken = np.zeros(M)
        np.add.at(taken, pos, inflow_S)
        S_env -= taken; S_env = np.maximum(S_env, 0.0)

        # --- Energetics + reactions ---
        E += seasonal_energy(t) + rng.normal(0,0.005,N)
        E = np.maximum(E,0.0)

        prod1, reac1 = X*W+1e-12, S_int*E+1e-12
        J1 = kf1*reac1 - kr1*prod1
        A1 = np.log(np.clip(reac1/prod1,1e-8,1e8)) + dG1/kT
        prod2, reac2 = (X*X*W)+1e-12, (X*E)+1e-12
        J2 = kf2*reac2 - kr2*prod2
        A2 = np.log(np.clip(reac2/prod2,1e-8,1e8)) + dG2/kT
        entropy_flux = np.sum(J1*A1 + J2*A2)
        sigma = np.abs(entropy_flux)

        # --- Update states ---
        S_int = np.maximum(S_int + inflow_S - J1, 0)
        X = np.maximum(X + J1 + J2 - cell_decay*X, 0)
        E = np.maximum(E - J1 - J2 - maint_cost - move_cost*np.abs(move), 0)
        W = np.maximum(W + J1 + J2 - 0.05*W, 0)

        # --- RL policy updates ---
        r_viab = -(X-1)**2
        r_ctrl = -lambda_ctrl*(a_u**2 + a_m**2)
        r = r_viab + r_ctrl - (lambda_diss*sigma)/N
        if not disable_rl:
            theta_u += eta*r[:,None]*(a_u*(1-a_u))[:,None]*feats
            theta_m += eta*r[:,None]*(1-a_m**2)[:,None]*feats
            theta_u = np.clip(theta_u, -theta_clip, theta_clip)
            theta_m = np.clip(theta_m, -theta_clip, theta_clip)

        # --- Death & reproduction ---
        for i in range(N):
            if X[i]<death_threshold_x or E[i]<death_threshold_e:
                deaths+=1
                j = np.argmax(X)
                S_int[i]=0.5*S_int[j]; X[i]=0.6*X[j]; E[i]=0.6*E[j]; W[i]=0.2*W[j]
                theta_u[i]=theta_u[j]+rng.normal(0,mut_scale,feat_dim)
                theta_m[i]=theta_m[j]+rng.normal(0,mut_scale,feat_dim)
                root_id[i]=root_id[j]; parent_id[i]=parent_id[j]; generation[i]=generation[j]
            elif X[i]>x_divide and E[i]>0.35:
                births+=1
                parent=i
                j=np.argmin(X)
                if j==parent and N>1:
                    j=np.argsort(X)[1]
                child=j
                S_int_child,X_child,E_child,W_child=0.6*S_int[parent],0.6*X[parent],0.6*E[parent],0.5*W[parent]
                S_int[parent]*=0.6; X[parent]*=0.6; E[parent]*=0.6; W[parent]*=0.5
                theta_u_child=theta_u[parent]+rng.normal(0,mut_scale,feat_dim)
                theta_m_child=theta_m[parent]+rng.normal(0,mut_scale,feat_dim)
                S_int[child],X[child],E[child],W[child]=S_int_child,X_child,E_child,W_child
                theta_u[child],theta_m[child]=theta_u_child,theta_m_child
                pos[child]=(pos[parent]+rng.integers(-1,2))%M
                lineage_log.append({"time":t,"child_id":int(next_id),"parent_id":int(agent_id[parent]),"generation":int(generation[parent]+1)})
                parent_id[child]=agent_id[parent]; agent_id[child]=next_id; next_id+=1; generation[child]=generation[parent]+1; root_id[child]=root_id[parent]

        # --- Logging ---
        if t%log_every==0:
            H_move = safe_entropy(a_m) if not disable_info else 0.0
            H_uptake = safe_entropy(a_u) if not disable_info else 0.0
            I_S_move = safe_mutual_info(S_local, a_m) if not disable_info else 0.0
            I_S_uptake = safe_mutual_info(S_local, a_u) if not disable_info else 0.0
            info_eff = (I_S_move + I_S_uptake)/(sigma+1e-12)
            KL_u = np.mean((theta_u - theta_prev_u)**2)
            KL_m = np.mean((theta_m - theta_prev_m)**2)
            theta_prev_u, theta_prev_m = theta_u.copy(), theta_m.copy()

            log.append({
                "time":t,"mean_X":np.mean(X),"mean_E":np.mean(E),
                "public_pool":P,"entropy_prod":sigma,"entropy_flux":entropy_flux,
                "H_move":H_move,"H_uptake":H_uptake,
                "I_S_move":I_S_move,"I_S_uptake":I_S_uptake,
                "info_efficiency":info_eff,
                "KL_theta_u":KL_u,"KL_theta_m":KL_m,
                "births":births,"deaths":deaths
            })

    return pd.DataFrame(log), pd.DataFrame(lineage_log)

# ============================================================
# 4️⃣ MULTI-SEED EXECUTION
# ============================================================

summary_records = []

for seed in tqdm(SEEDS, desc="Running multi-seed simulations"):
    df_log, df_lin = run_protocell_simulation(seed=seed)
    df_log["seed"] = seed
    df_lin["seed"] = seed
    df_log.to_csv(f"results/log_seed{seed}.csv", index=False)
    df_lin.to_csv(f"results/lineage_seed{seed}.csv", index=False)

    summary_records.append({
        "seed": seed,
        "mean_X_final": df_log["mean_X"].iloc[-1],
        "entropy_mean": df_log["entropy_prod"].mean(),
        "info_eff_mean": df_log["info_efficiency"].mean(),
        "KL_u_mean": df_log["KL_theta_u"].mean(),
        "KL_m_mean": df_log["KL_theta_m"].mean()
    })

summary = pd.DataFrame(summary_records)
summary.to_csv("results/summary_all.csv", index=False)

# ============================================================
# 5️⃣ STATISTICS & FIGURES
# ============================================================

plt.figure(figsize=(12,6))
for seed in SEEDS:
    df = pd.read_csv(f"results/log_seed{seed}.csv")
    plt.plot(df["time"], df["mean_X"], label=f"Seed {seed}", alpha=0.7)
plt.xlabel("Time"); plt.ylabel("⟨X⟩"); plt.title("Learning Dynamics Across Seeds")
plt.legend(); plt.tight_layout(); plt.savefig("results/mean_X_allseeds.png", dpi=300); plt.close()

plt.figure(figsize=(12,6))
sns.barplot(data=summary, x="seed", y="info_eff_mean")
plt.title("Information–Dissipation Efficiency (mean per seed)")
plt.tight_layout(); plt.savefig("results/info_efficiency_summary.png", dpi=300); plt.close()

# ============================================================
# 6️⃣ METADATA
# ============================================================
meta = {
    "DISABLE_RL": disable_rl, "DISABLE_INFO": disable_info,
    "GAMMA": gamma, "C_SHARE": c_share,
    "LAMBDA_DISS": lambda_diss, "ENV_NOISE": env_noise,
    "N": N, "M": M, "T": T, "SEEDS": len(SEEDS)
}
pd.DataFrame([meta]).to_csv("results/run_metadata.csv", index=False)

print("\n✅ Completed all protocell simulations.")
print("Saved summary: results/summary_all.csv")
print("Saved figures: results/mean_X_allseeds.png, info_efficiency_summary.png")
