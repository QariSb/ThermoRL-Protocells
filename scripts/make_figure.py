#!/usr/bin/env python3
# ============================================================
# Protocell Figure Generator — Extended Metrics Edition
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

sns.set_context("talk")
sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 11})
os.makedirs("results", exist_ok=True)

# ============================================================
# Load Data
# ============================================================
log_files = [f for f in os.listdir("results") if f.startswith("log_seed") and f.endswith(".csv")]
logs = []
for f in log_files:
    df = pd.read_csv(os.path.join("results", f))
    seed = int(f.split("seed")[1].split(".")[0])
    df["seed"] = seed
    logs.append(df)
df_all = pd.concat(logs, ignore_index=True)

# ============================================================
# Derived Metrics (fixed coupling_XE issue)
# ============================================================
df_all["dXdt"] = df_all.groupby("seed")["mean_X"].diff().fillna(0)
df_all["learning_efficiency"] = df_all["dXdt"] / (df_all["entropy_prod"] + 1e-12)
df_all["entropy_per_flux"] = np.abs(df_all["entropy_prod"] / (df_all["dXdt"].abs() + 1e-12))
df_all["policy_variance_total"] = df_all["KL_theta_u"] + df_all["KL_theta_m"]
df_all["info_efficiency"] = (df_all["I_S_move"] + df_all["I_S_uptake"]) / (df_all["entropy_prod"] + 1e-12)

# --- NEW: X–E coupling (per-seed correlation coefficient)
coupling_vals = []
for seed, group in df_all.groupby("seed"):
    if group["mean_X"].std() > 1e-9 and group["mean_E"].std() > 1e-9:
        r, _ = pearsonr(group["mean_X"], group["mean_E"])
    else:
        r = np.nan
    df_all.loc[group.index, "coupling_XE"] = r
    coupling_vals.append({"seed": seed, "coupling_XE": r})
df_coupling = pd.DataFrame(coupling_vals)

# --- Aggregate summary
summary = (
    df_all.groupby("seed")
    .agg({
        "mean_X": "last",
        "entropy_prod": "mean",
        "learning_efficiency": "mean",
        "entropy_per_flux": "mean",
        "policy_variance_total": "mean",
        "info_efficiency": "mean"
    })
    .reset_index()
    .merge(df_coupling, on="seed", how="left")
)
summary.to_csv("results/summary_extended.csv", index=False)


# ============================================================
# Figure 1 — Reinforcement Learning & Convergence (4 panels)
# ============================================================
plt.figure(figsize=(11,9))

# (a) Mean X ± std across seeds
plt.subplot(2,2,1)
agg = df_all.groupby("time")["mean_X"].agg(["mean","std"]).reset_index()
plt.plot(agg["time"], agg["mean"], color="teal", label="Mean ⟨X⟩")
plt.fill_between(agg["time"], agg["mean"]-agg["std"], agg["mean"]+agg["std"],
                 color="teal", alpha=0.2, label="±1 SD")
plt.xlabel("Time"); plt.ylabel("⟨X⟩")
plt.title("Policy Reinforcement (mean ± std)")
plt.legend()

# (b) Policy Diversity (KL drift ± std)
plt.subplot(2,2,2)
agg_u = df_all.groupby("time")["KL_theta_u"].agg(["mean","std"]).reset_index()
agg_m = df_all.groupby("time")["KL_theta_m"].agg(["mean","std"]).reset_index()
plt.plot(agg_u["time"], agg_u["mean"], color="tab:blue", label="θ_u mean")
plt.fill_between(agg_u["time"], agg_u["mean"]-agg_u["std"], agg_u["mean"]+agg_u["std"],
                 color="tab:blue", alpha=0.2)
plt.plot(agg_m["time"], agg_m["mean"], color="tab:orange", label="θ_m mean")
plt.fill_between(agg_m["time"], agg_m["mean"]-agg_m["std"], agg_m["mean"]+agg_m["std"],
                 color="tab:orange", alpha=0.2)
plt.xlabel("Time"); plt.ylabel("KL variance")
plt.title("Policy Diversity (mean ± std)")
plt.legend()

# (c) Internal Variable Convergence (S_int)
plt.subplot(2,2,3)
if "mean_Sint" in df_all.columns:
    agg_s = df_all.groupby("time")["mean_Sint"].agg(["mean","std"]).reset_index()
    plt.plot(agg_s["time"], agg_s["mean"], color="navy")
    plt.fill_between(agg_s["time"], agg_s["mean"]-agg_s["std"], agg_s["mean"]+agg_s["std"],
                     color="navy", alpha=0.25)
    plt.axhline(1.0, ls="--", c="black", label="Target x*")
    plt.xlabel("Time"); plt.ylabel("⟨S_int⟩")
    plt.title("Internal Variable Convergence (mean ± std)")
else:
    plt.text(0.3,0.5,"No mean_Sint logged",ha="center",va="center"); plt.axis("off")

# (d) Learned Policy Space (θ_u vs θ_m, all seeds)
plt.subplot(2,2,4)
theta_path = "results/final_theta.csv"
if os.path.exists(theta_path):
    th = pd.read_csv(theta_path)
    sns.scatterplot(x="mean_theta_u", y="mean_theta_m", hue="generation",
                    data=th, palette="viridis", s=60)
    plt.xlabel("Mean θ_u"); plt.ylabel("Mean θ_m")
    plt.title("Learned Policy Space (color = generation)")
else:
    plt.text(0.3,0.5,"final_theta.csv not found",ha="center",va="center"); plt.axis("off")

plt.suptitle("Reinforcement Learning & Convergence", fontsize=16, fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig("results/Fig1_RL_Convergence_Multiseed.png", dpi=300)
plt.close()


# ============================================================
# Figure 2 — Thermodynamics & Environment (3 rows, mean ± std)
# ============================================================
plt.figure(figsize=(10,8))
metrics = ["entropy_prod","mean_E","public_pool"]
colors = ["purple","red","green"]
titles = ["Entropy Production (Σ)","Mean Energy ⟨E⟩","Public Resource Pool (P)"]

for i,(m,c,t) in enumerate(zip(metrics,colors,titles),start=1):
    plt.subplot(3,1,i)
    agg = df_all.groupby("time")[m].agg(["mean","std"]).reset_index()
    plt.plot(agg["time"], agg["mean"], color=c)
    plt.fill_between(agg["time"], agg["mean"]-agg["std"], agg["mean"]+agg["std"],
                     color=c, alpha=0.25)
    plt.title(t); plt.ylabel(m); 
    if i==3: plt.xlabel("Time")

plt.tight_layout()
plt.savefig("results/Fig2_Thermodynamics_Multiseed.png", dpi=300)
plt.close()


# ============================================================
# Figure 3 — Autocatalysis & Chemical Coupling
# ============================================================
plt.figure(figsize=(9,6))
agg_dx = df_all.groupby("time")["dXdt"].agg(["mean","std"]).reset_index()

plt.subplot(2,2,1)
plt.plot(agg_dx["time"], agg_dx["mean"], color="magenta")
plt.fill_between(agg_dx["time"], agg_dx["mean"]-agg_dx["std"], agg_dx["mean"]+agg_dx["std"],
                 color="magenta", alpha=0.25)
plt.title("Autocatalytic Flux (⟨dX/dt⟩ ± std)")
plt.xlabel("Time"); plt.ylabel("Flux")

plt.subplot(2,2,2)
sns.scatterplot(data=df_all.sample(frac=0.25, random_state=1),
                x="dXdt", y="entropy_prod", alpha=0.4, color="purple")
plt.title("Entropy vs Flux (Hysteresis Loop)")
plt.xlabel("Flux"); plt.ylabel("Σ")

plt.subplot(2,2,3)
sns.scatterplot(data=df_all.sample(frac=0.25, random_state=2),
                x="mean_X", y="mean_E", alpha=0.5, color="blue")
plt.title("Chemical Coupling X–E (all seeds)")
plt.xlabel("⟨X⟩"); plt.ylabel("⟨E⟩")

plt.subplot(2,2,4)
sns.scatterplot(data=df_all.sample(frac=0.25, random_state=3),
                x="mean_X", y="entropy_per_flux", alpha=0.5, color="orange")
plt.title("Entropy per Flux Efficiency (Σ/|dX/dt|)")
plt.xlabel("⟨X⟩"); plt.ylabel("Σ/|dX/dt|")

plt.tight_layout()
plt.savefig("results/Fig3_Autocatalysis_Multiseed.png", dpi=300)
plt.close()


# ============================================================
# Figure 4 — Evolution & Lineages (aggregated, all seeds)
# ============================================================

# Load lineage CSVs (if present)
lineage_files = [f for f in os.listdir("results") if f.startswith("lineage_seed") and f.endswith(".csv")]
if lineage_files:
    lineages = [pd.read_csv(os.path.join("results", f)) for f in lineage_files]
    df_lin = pd.concat(lineages, ignore_index=True)
else:
    df_lin = pd.DataFrame(columns=["time","generation"])  # empty fallback

plt.figure(figsize=(10,8))

# (a) Births vs deaths (mean ± std across seeds)
plt.subplot(2,2,1)
agg_birth = df_all.groupby("time")["births"].agg(["mean","std"]).reset_index()
agg_death = df_all.groupby("time")["deaths"].agg(["mean","std"]).reset_index()
plt.plot(agg_birth["time"], agg_birth["mean"], color="green", label="Births")
plt.fill_between(agg_birth["time"], agg_birth["mean"]-agg_birth["std"], agg_birth["mean"]+agg_birth["std"],
                 color="green", alpha=0.25)
plt.plot(agg_death["time"], agg_death["mean"], color="red", label="Deaths")
plt.fill_between(agg_death["time"], agg_death["mean"]-agg_death["std"], agg_death["mean"]+agg_death["std"],
                 color="red", alpha=0.25)
plt.title("Births vs Deaths (mean ± std)"); plt.xlabel("Time"); plt.ylabel("Count"); plt.legend()

# (b) Lineage depth over time (all seeds)
plt.subplot(2,2,2)
if not df_lin.empty:
    sns.scatterplot(data=df_lin, x="time", y="generation", s=10, alpha=0.5)
    plt.title("Lineage Depth over Time (all seeds)")
    plt.xlabel("Time"); plt.ylabel("Generation")
else:
    plt.text(0.3,0.5,"No lineage files found",ha="center",va="center"); plt.axis("off")

# (c) Clade proliferation histogram
plt.subplot(2,2,4)
if not df_lin.empty:
    sns.histplot(df_lin["generation"], bins=12, color="olive")
    plt.title("Clade Proliferation by Generation (all seeds)")
    plt.xlabel("Generation"); plt.ylabel("Count")
else:
    plt.text(0.3,0.5,"No lineage data",ha="center",va="center"); plt.axis("off")

plt.suptitle("Evolution & Lineages", fontsize=16, fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig("results/Fig4_Biology_Multiseed.png", dpi=300)
plt.close()



# ============================================================
# Figure 5 — Information Theory Metrics (mean ± std)
# ============================================================
plt.figure(figsize=(9,10))
metrics = [
    ("I_S_move","blue","Mutual Info (S↔A)"),
    ("info_efficiency","green","Information Efficiency (I/Σ)"),
    ("entropy_prod","maroon","Thermodynamic Entropy (Σ)"),
    ("learning_efficiency","teal","Learning Efficiency (Δ⟨X⟩/Σ)"),
    ("policy_variance_total","darkorange","Total Policy Variance (Var θ)")
]

for i,(m,c,t) in enumerate(metrics, start=1):
    plt.subplot(len(metrics),1,i)
    agg = df_all.groupby("time")[m].agg(["mean","std"]).reset_index()
    plt.plot(agg["time"], agg["mean"], color=c)
    plt.fill_between(agg["time"], agg["mean"]-agg["std"], agg["mean"]+agg["std"],
                     color=c, alpha=0.25)
    plt.ylabel(m); plt.title(t)
    if i==len(metrics): plt.xlabel("Time")

plt.suptitle("Information Metrics (mean ± std across seeds)", fontsize=16, fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig("results/Fig5_InfoTheory_Multiseed.png", dpi=300)
plt.close()

# ============================================================
# Summary
# ============================================================
print("✅ Generated extended metrics and 5 upgraded figures:")
print("  - Fig1_RL_Convergence_Extended.png")
print("  - Fig2_Thermodynamics_Extended.png")
print("  - Fig3_Autocatalysis_Extended.png")
print("  - Fig4_Biology_Extended.png")
print("  - Fig5_InformationTheory_Extended.png")
print("Additional summary: results/summary_extended.csv")
