#!/usr/bin/env python3
# ============================================================
# Protocell Master Figures
# Generates four multi-panel figures summarizing Reinforcement Learning,
# Physics, Chemistry, and Biology aspects from the protocell simulation.
# Requires:
#   results/repro_visible_with_lineage_summary.csv
#   results/lineage_table.csv
# Outputs:
#   results/fig1_rl_convergence.png
#   results/fig2_physics.png
#   results/fig3_chemistry.png
#   results/fig4_biology.png
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import networkx as nx
    HAS_NX = True
except Exception:
    HAS_NX = False

SUMMARY_PATH = "results/repro_visible_with_lineage_summary.csv"
LINEAGE_PATH = "results/lineage_table.csv"

os.makedirs("results", exist_ok=True)
if not os.path.exists(SUMMARY_PATH):
    raise FileNotFoundError("Run protocell_simulation.py first to produce results.")

df = pd.read_csv(SUMMARY_PATH)
if os.path.exists(LINEAGE_PATH):
    lineage = pd.read_csv(LINEAGE_PATH)
else:
    lineage = pd.DataFrame()

# ------------------------------------------------------------
# Figure 1: Reinforcement Learning and Convergence
# ------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs = axs.ravel()

# (A) Mean reward over time
axs[0].plot(df["time"], df["mean_X"], color="teal", label="Mean X (proxy reward)")
axs[0].set_title("Policy Reinforcement and Viability Proxy")
axs[0].set_xlabel("Time"); axs[0].set_ylabel("Mean X"); axs[0].legend()

# (B) Policy diversity
axs[1].plot(df["time"], df["theta_u_var"], label="Var θ_u")
axs[1].plot(df["time"], df["theta_m_var"], label="Var θ_m")
axs[1].set_title("Policy Diversity (Exploration)")
axs[1].set_xlabel("Time"); axs[1].set_ylabel("Variance")
axs[1].legend()

# (C) Internal variable convergence
axs[2].plot(df["time"], df["mean_Sint"], color="darkblue")
axs[2].axhline(1.0, ls="--", c="k", lw=1, label="Target x*")
axs[2].set_title("Internal Variable Convergence (Sint)")
axs[2].set_xlabel("Time"); axs[2].set_ylabel("Mean S_int"); axs[2].legend()

# (D) Placeholder for θ scatter (requires later snapshot data)
#axs[3].text(0.5, 0.5, "θ_u vs θ_m scatter\n(add from snapshot)", ha="center", va="center")
#axs[3].set_title("Learned Parameter Space")

theta = pd.read_csv("results/final_theta.csv")
axs[3].scatter(theta["mean_theta_u"], theta["mean_theta_m"],
               c=theta["generation"], cmap="viridis", s=50)
axs[3].set_xlabel("Mean θ_u"); axs[3].set_ylabel("Mean θ_m")
axs[3].set_title("Learned Policy Space (color = generation)")


fig.suptitle("Figure 1 — Reinforcement Learning & Convergence", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("results/fig1_rl_convergence.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# Figure 2: Physics (Thermodynamics & Environment)
# ------------------------------------------------------------
fig, axs = plt.subplots(3, 2, figsize=(12, 9))
axs = axs.ravel()

# (A) Resource
axs[0].plot(df["time"], df["resource_total"], color="orange")
axs[0].set_title("Total Environmental Resource")
axs[0].set_xlabel("Time"); axs[0].set_ylabel("R_total")

# (B) Energy
axs[1].plot(df["time"], df["mean_E"], color="red")
axs[1].set_title("Mean Energy (E)")
axs[1].set_xlabel("Time"); axs[1].set_ylabel("Mean E")

# (C) Entropy production
axs[2].plot(df["time"], df["entropy_prod"], color="purple")
axs[2].set_title("Entropy Production (Σ)")
axs[2].set_xlabel("Time"); axs[2].set_ylabel("Σ")

# (D) Gini (inequality proxy — not stored, we simulate smooth version)
axs[3].plot(df["time"], np.gradient(df["resource_total"]), color="brown")
axs[3].set_title("Resource Flux Gradient (proxy inequality)")
axs[3].set_xlabel("Time"); axs[3].set_ylabel("ΔR/Δt")

# (E) Public pool
axs[4].plot(df["time"], df["public_pool"], color="green")
axs[4].set_title("Public Resource Pool (P)")
axs[4].set_xlabel("Time"); axs[4].set_ylabel("P")

# (F) Empty slot
axs[5].axis("off")

fig.suptitle("Figure 2 — Physics: Thermodynamics & Environment", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("results/fig2_physics.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# Figure 3: Chemistry (Reactions & Autocatalysis)
# ------------------------------------------------------------
# We'll derive proxy flux and affinity trends from entropy production (Σ) and mean_X dynamics
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs = axs.ravel()

# (A) Proxy for J2 (autocatalytic flux) from dX/dt
J2_proxy = np.gradient(df["mean_X"])
axs[0].plot(df["time"], J2_proxy, color="magenta")
axs[0].set_title("Autocatalytic Flux Proxy (dX/dt)")
axs[0].set_xlabel("Time"); axs[0].set_ylabel("Flux")

# (B) Entropy vs. flux
axs[1].scatter(J2_proxy, df["entropy_prod"], s=10, alpha=0.5, color="purple")
axs[1].set_title("Entropy vs. Autocatalytic Flux (Hysteresis Loop)")
axs[1].set_xlabel("Flux Proxy"); axs[1].set_ylabel("Entropy Σ")

# (C) Mean X vs. mean E (reaction coupling)
axs[2].scatter(df["mean_X"], df["mean_E"], s=10, alpha=0.5, color="blue")
axs[2].set_title("Chemical Coupling: X–E Phase Portrait")
axs[2].set_xlabel("Mean X"); axs[2].set_ylabel("Mean E")

# (D) Placeholder for stoichiometric schematic
axs[3].text(0.5, 0.5, "Reaction network schematic\n(R1: S+E↔X+W, R2: X+E↔2X+W)", ha="center", va="center")
axs[3].set_axis_off()

fig.suptitle("Figure 3 — Chemistry: Autocatalysis & Reactions", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("results/fig3_chemistry.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# Figure 4: Biology (Evolution & Lineages)
# ------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs = axs.ravel()

# (A) Births vs deaths
axs[0].plot(df["time"], df["births"], label="Births", color="green")
axs[0].plot(df["time"], df["deaths"], label="Deaths", color="red")
axs[0].set_title("Births vs Deaths")
axs[0].set_xlabel("Time"); axs[0].set_ylabel("Count"); axs[0].legend()

# (B) Mean generation depth proxy
if not lineage.empty:
    axs[1].plot(lineage["time"], lineage["generation"], ".", alpha=0.5, color="blue")
    axs[1].set_title("Lineage Generation Depth over Time")
    axs[1].set_xlabel("Time"); axs[1].set_ylabel("Generation")
else:
    axs[1].text(0.5, 0.5, "No lineage data", ha="center", va="center")

# (C) Radial lineage tree
if HAS_NX and not lineage.empty:
    import networkx as nx
    G = nx.DiGraph()
    for _, row in lineage.iterrows():
        G.add_edge(int(row["parent_id"]), int(row["child_id"]))

    # Build generation map including parents (roots = 0)
    gen_map = {}
    for _, row in lineage.iterrows():
        p, c, g = int(row["parent_id"]), int(row["child_id"]), int(row["generation"])
        gen_map[c] = g
        if p not in gen_map:
            gen_map[p] = 0  # assign 0 to unseen parents

    # Ensure all graph nodes have a generation entry
    for n in G.nodes():
        gen_map.setdefault(n, 0)

    gens = list(gen_map.values())
    min_g, max_g = min(gens), max(gens)



    
    shells = [[n for n, gg in gen_map.items() if gg == g] for g in range(min_g, max_g+1)]
    shells = [s for s in shells if len(s) > 0]
    try:
        pos = nx.shell_layout(G, nlist=shells)
    except Exception:
        pos = nx.spring_layout(G, seed=0)
    node_colors = [gen_map[n] for n in G.nodes()]
    nx.draw_networkx_edges(G, pos, ax=axs[2], alpha=0.3, arrows=False)
    nx.draw_networkx_nodes(G, pos, ax=axs[2], node_color=node_colors, cmap="viridis", node_size=20)
    axs[2].set_title("Lineage Tree (Radial shells)"); axs[2].axis("off")
else:
    axs[2].text(0.5, 0.5, "Lineage tree unavailable", ha="center", va="center"); axs[2].set_axis_off()

# (D) Clade abundance / fitness
if not lineage.empty:
    clade_counts = lineage.groupby("generation").size()
    axs[3].bar(clade_counts.index, clade_counts.values, color="olive")
    axs[3].set_title("Clade Proliferation by Generation")
    axs[3].set_xlabel("Generation"); axs[3].set_ylabel("Count")
else:
    axs[3].set_axis_off()

fig.suptitle("Figure 4 — Biology: Evolution & Lineages", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("results/fig4_biology.png", dpi=300)
plt.close()

print("All master figures generated and saved in results/:")
for f in ["fig1_rl_convergence.png","fig2_physics.png","fig3_chemistry.png","fig4_biology.png"]:
    print(" -", os.path.join("results", f))
