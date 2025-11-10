# ThermoRL-Protocells

**Thermodynamic Reinforcement Learning in Artificial Protocells**  
_A project by Abdul Basit (2025)_

---

## 🧬 Overview

This repository explores the **emergence of life-like behavior** in artificial protocells using **multi-agent reinforcement learning** constrained by **thermodynamic and chemical principles**.

Protocells act as agents that learn to regulate internal chemistry, share environmental resources, and reproduce adaptively.  
The system exhibits **autocatalytic dynamics**, **energy dissipation**, and **lineage diversification**—mirroring physical and biological aspects of living systems.

---

## 🔬 Core Features

- 🧠 **Reinforcement Learning**: Multi-agent policy adaptation via local feedback.  
- ⚙️ **Thermodynamic Environment**: Resource exchange and entropy production.  
- ⚗️ **Autocatalytic Chemistry**: Reversible reactions driving metabolic feedback.  
- 🧩 **Evolutionary Dynamics**: Mutation, reproduction, and lineage tracking.  
- 📊 **Integrated Visualization**: Four composite figures summarizing emergent behavior.

---

## 🧠 Scientific Structure

| Figure | Theme | Description |
|---------|--------|-------------|
| **Figure 1** | *Reinforcement Learning & Convergence* | Shows learning dynamics, policy diversity, and internal homeostasis. |
| **Figure 2** | *Physics* | Energy flow, entropy production, and environmental fluctuations. |
| **Figure 3** | *Chemistry* | Autocatalytic reaction fluxes and phase portraits. |
| **Figure 4** | *Biology* | Reproduction, lineage trees, and ecological turnover. |

---

## 🧰 Usage

### 1️⃣ Run the simulation

```bash
python simulation/protocell_simulation.py
```

This produces raw data in `results/` including:
- `repro_visible_with_lineage_summary.csv`
- `lineage_table.csv`

### 2️⃣ Generate figures

```bash
python simulation/protocell_master_figures.py
```

This generates four integrated multi-panel figures:

```
results/
 ├─ fig1_rl_convergence.png
 ├─ fig2_physics.png
 ├─ fig3_chemistry.png
 └─ fig4_biology.png
```

---

## 📂 Repository Structure

```
ThermoRL-Protocells/
├── README.md
├── simulation/
│   ├── protocell_simulation.py
│   ├── protocell_master_figures.py
│   ├── protocell_plots.py
│   └── configs/
├── results/
├── figures/
└── docs/
```

---

## 📊 Example Output

![Figure 1](figures/fig1_rl_convergence.png)

**Interpretation:**  
Agents exhibit adaptive oscillations in internal state and ongoing diversity in policy parameters—hallmarks of open-ended evolution under energy flow.

---

## 🧩 Citation

If you use this framework in academic work:

> Basit, A. (2025). *Thermodynamic Reinforcement Learning in Artificial Protocells*.

---

## 📄 License

MIT License © 2025 Abdul Basit  
For research and educational use only.
