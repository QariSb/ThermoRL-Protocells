# RL-favoring patched simulation script
# ------------------------------------------------------------
# This script implements:
# - Predictive cues with pulse_timer
# - Episodic reproduction rewards
# - Feature memory traces
# - Slow genetic evolution
# - Higher learning rate for RL
# - Advantage-based updates
# - Vacancy-only reproduction
# - Donor cost
# - Paired seeds for RL_ON vs RL_OFF
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("pilot_patch_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def patched_sim(seed, RL=True, T=1500, N=12, M=24):
    rng = np.random.default_rng(seed)

    S_env = np.full(M, 30.0)
    pos = rng.integers(0, M, size=N)

    S_int = rng.uniform(0.25, 0.5, N)
    X = rng.uniform(0.4, 0.6, N)
    E = rng.uniform(0.4, 0.6, N)

    feat_dim = 6
    theta = rng.normal(0, 0.08, (N, feat_dim))
    mut_scale = 0.005
    eta = 0.03 if RL else 0.0
    theta_clip = 4.0

    feat_mem = np.zeros((N, feat_dim))
    alpha_mem = 0.06

    pulse_timer = np.zeros(M, dtype=int)
    pulse_delay_mean = 6
    pulse_strength_low, pulse_strength_high = 6.0, 14.0
    pulse_schedule_prob = 0.02

    alive = np.ones(N, dtype=bool)
    vac = np.zeros(N, dtype=int)
    vacancy_min, vacancy_max = 3, 12
    donor_cost_frac = 0.12

    ep_reward = np.zeros(N)
    episodic_update_period = 12

    logs = []

    for t in range(T):
        if rng.random() < pulse_schedule_prob:
            mask = (rng.random(M) < 0.14)
            delays = rng.integers(max(2, pulse_delay_mean - 2), pulse_delay_mean + 3, size=M)
            pulse_timer[mask] = delays[mask]

        pulse_timer = np.maximum(pulse_timer - 1, 0)
        occurring = (pulse_timer == 1)
        if occurring.any():
            strength = rng.uniform(pulse_strength_low, pulse_strength_high, size=M)
            S_env[occurring] += strength[occurring]
            pulse_timer[occurring] = 0

        S_env += rng.normal(0, 0.12, M)
        S_env = np.maximum(S_env, 0.0)

        cue = (pulse_timer > 0).astype(float)
        S_local = S_env[pos] + rng.normal(0, 0.12, size=N)

        feats = np.stack([S_int, X, E, S_local, cue[pos], np.zeros(N)], axis=1)

        feat_mem = (1 - alpha_mem) * feat_mem + alpha_mem * feats
        used_feats = feat_mem.copy()

        z = np.sum(theta * used_feats, axis=1)
        z = np.clip(z, -30, 30)
        a = 1.0 / (1.0 + np.exp(-z))

        uptake = a * 0.18
        uptake = np.minimum(uptake, S_env[pos])
        S_int += uptake
        S_env[pos] -= uptake

        X = np.maximum(X + 0.06 * S_int - 0.014 * X, 0.0)
        E = np.maximum(E + 0.022 - 0.015, 0.0)

        # Death handling
        for i in range(N):
            if alive[i] and (X[i] < 0.05 or E[i] < 0.03):
                alive[i] = False
                vac[i] = rng.integers(vacancy_min, vacancy_max + 1)
                ep_reward[i] -= 4.0  # episodic death penalty

        vac = np.maximum(vac - 1, 0)

        # Division to fill vacancies only
        for i in range(N):
            if not alive[i]:
                continue

            if X[i] > 0.82 and E[i] > 0.18:
                vacancies = np.where((~alive) & (vac > 0))[0]

                if vacancies.size > 0:
                    child = int(vacancies[0])

                    # donor cost adjustment
                    S_int_child = 0.5 * S_int[i]
                    S_int[i] *= 0.5 * (1.0 - donor_cost_frac)

                    X_child = 0.5 * X[i] * (1.0 - donor_cost_frac)
                    X[i] *= 0.5 * (1.0 - donor_cost_frac)

                    E_child = 0.5 * E[i] * (1.0 - donor_cost_frac)
                    E[i] *= 0.5 * (1.0 - donor_cost_frac)

                    S_int[child] = S_int_child
                    X[child] = X_child
                    E[child] = E_child

                    theta[child] = theta[i] + rng.normal(0, mut_scale, feat_dim)

                    alive[child] = True
                    vac[child] = 0

                    ep_reward[i] += 6.0  # reward successful reproduction

        # Base reward dynamics
        base_r = - (X - 0.9)**2
        ep_reward += base_r * 0.02

        # Episodic RL updates
        if RL and (t % episodic_update_period == 0):
            alive_idx = np.where(alive)[0]

            if alive_idx.size > 0:
                baseline = np.mean(ep_reward[alive_idx])
                adv = ep_reward - baseline

                theta[alive_idx] += eta * adv[alive_idx, None] *                                     (a * (1 - a))[alive_idx, None] * used_feats[alive_idx]

                ep_reward[alive_idx] *= 0.25

        # Life-long incremental updates
        if RL:
            alive_idx = np.where(alive)[0]

            if alive_idx.size > 0:
                r = base_r
                mean_r = np.mean(r[alive_idx])
                std_r = np.std(r[alive_idx]) + 1e-8

                r_norm = (r - mean_r) / std_r

                theta[alive_idx] += (eta * 0.25) * r_norm[alive_idx, None] * used_feats[alive_idx]

        # Clip policy parameters
        theta = np.clip(theta, -theta_clip, theta_clip)

        logs.append({
            "t": t,
            "meanX": float(np.mean(X[alive]) if np.any(alive) else 0.0),
            "alive": int(np.sum(alive))
        })

    return pd.DataFrame(logs), theta, alive


# ------------------------------------------------------------
# Run pilot across 10 paired seeds
# ------------------------------------------------------------

NUM_SEEDS = 10
T = 1500

results_on, results_off = [], []

for s in range(NUM_SEEDS):
    seed0 = 5000 + s
    df_on, _, _ = patched_sim(seed0, RL=True, T=T)
    df_off, _, _ = patched_sim(seed0, RL=False, T=T)

    results_on.append(df_on)
    results_off.append(df_off)

# Aggregate
times = results_on[0]['t'].values
meanX_on = np.vstack([df['meanX'].values for df in results_on])
meanX_off = np.vstack([df['meanX'].values for df in results_off])

alive_on = np.vstack([df['alive'].values for df in results_on])
alive_off = np.vstack([df['alive'].values for df in results_off])

mean_on = meanX_on.mean(axis=0)
sem_on = meanX_on.std(axis=0) / np.sqrt(NUM_SEEDS)

mean_off = meanX_off.mean(axis=0)
sem_off = meanX_off.std(axis=0) / np.sqrt(NUM_SEEDS)

alive_mean_on = alive_on.mean(axis=0)
alive_sem_on = alive_on.std(axis=0) / np.sqrt(NUM_SEEDS)

alive_mean_off = alive_off.mean(axis=0)
alive_sem_off = alive_off.std(axis=0) / np.sqrt(NUM_SEEDS)

# ------------------------------------------------------------
# Save aggregated CSV summary
# ------------------------------------------------------------

agg_rows = []
for s in range(NUM_SEEDS):
    agg_rows.append({
        "condition": "RL_ON",
        "seed": s,
        "final_meanX": float(results_on[s]['meanX'].iloc[-1]),
        "final_alive": int(results_on[s]['alive'].iloc[-1])
    })
    agg_rows.append({
        "condition": "RL_OFF",
        "seed": s,
        "final_meanX": float(results_off[s]['meanX'].iloc[-1]),
        "final_alive": int(results_off[s]['alive'].iloc[-1])
    })

agg_df = pd.DataFrame(agg_rows)
agg_df.to_csv(OUT_DIR / "patched_pilot_summary_across_seeds.csv", index=False)


# ------------------------------------------------------------
# Plot meanX with 95% CI
# ------------------------------------------------------------

plt.figure(figsize=(9, 4))
plt.plot(times, mean_on, label="RL_ON")
plt.fill_between(times, mean_on - 1.96 * sem_on, mean_on + 1.96 * sem_on, alpha=0.25)

plt.plot(times, mean_off, label="RL_OFF")
plt.fill_between(times, mean_off - 1.96 * sem_off, mean_off + 1.96 * sem_off, alpha=0.25)

plt.xlabel("time")
plt.ylabel("Mean X")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "patched_meanX.png", dpi=200)
plt.close()


# ------------------------------------------------------------
# Plot alive (occupied slots) with 95% CI
# ------------------------------------------------------------

plt.figure(figsize=(9, 4))
plt.plot(times, alive_mean_on, label="RL_ON")
plt.fill_between(times, alive_mean_on - 1.96 * alive_sem_on,
                 alive_mean_on + 1.96 * alive_sem_on, alpha=0.25)

plt.plot(times, alive_mean_off, label="RL_OFF")
plt.fill_between(times, alive_mean_off - 1.96 * alive_sem_off,
                 alive_mean_off + 1.96 * alive_sem_off, alpha=0.25)

plt.xlabel("time")
plt.ylabel("Occupied slots")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "patched_alive.png", dpi=200)
plt.close()


# ------------------------------------------------------------
# Save per-seed time-series logs
# ------------------------------------------------------------

for i in range(NUM_SEEDS):
    results_on[i].to_csv(OUT_DIR / f"RL_ON_seed_{i}.csv", index=False)
    results_off[i].to_csv(OUT_DIR / f"RL_OFF_seed_{i}.csv", index=False)


# ------------------------------------------------------------
# Paired statistics for RL advantage
# ------------------------------------------------------------

final_on = np.array([results_on[s]['meanX'].iloc[-1] for s in range(NUM_SEEDS)])
final_off = np.array([results_off[s]['meanX'].iloc[-1] for s in range(NUM_SEEDS)])

diff = final_on - final_off

mean_diff = float(diff.mean())
std_diff = float(diff.std(ddof=1))
t_stat = float(mean_diff / (std_diff / np.sqrt(NUM_SEEDS)))

# Bootstrap p-value estimate: P(mean_diff <= 0)
B = 10000
rng_boot = np.random.default_rng(244)
boot_means = []

for _ in range(B):
    idx = rng_boot.integers(0, NUM_SEEDS, NUM_SEEDS)
    boot_means.append(diff[idx].mean())

boot_means = np.array(boot_means)
p_boot = float(np.mean(boot_means <= 0.0))

stats_df = pd.DataFrame({
    "mean_diff": [mean_diff],
    "std_diff": [std_diff],
    "t_stat": [t_stat],
    "bootstrap_p": [p_boot],
    "seeds": [NUM_SEEDS]
})

stats_df.to_csv(OUT_DIR / "patched_stats.csv", index=False)


print("Patched pilot completed successfully.")
print("Results saved in:", OUT_DIR)
