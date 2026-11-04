# Attention-Based Deep Reinforcement Learning for Smart Lithium-Ion Battery Charging

**Research implementation from Arian Abdollahi's thesis and uploaded Kaggle notebook.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Learning](https://img.shields.io/badge/RL-Soft%20Actor--Critic-7755bb)](https://stable-baselines3.readthedocs.io/)
[![Environment](https://img.shields.io/badge/Environment-Gymnasium-208c57)](https://gymnasium.farama.org/)
[![Source](https://img.shields.io/badge/Source-Author%20notebook-007acc)](notebooks/original_wattention.ipynb)

> **Thesis:** Arian Abdollahi, *Dynamic Optimization of Smart Charging of Lithium-Ion Batteries Using Advanced Deep Learning Approaches*, Amirkabir University of Technology, Department of Electrical Engineering (Control Engineering), 2026 (Bahman 1404). Supervisor: Dr. Farzaneh Abdollahi.
>
> **Scope:** Simulation research only; not a real-world charger or validated battery management system.

## Overview

This project investigates **dynamic charging and balancing of a four-cell lithium-ion battery pack** using continuous-action deep reinforcement learning. The charging controller jointly selects **one normalized charging-current command per cell** and **one coolant-temperature control command**, while considering charging progress, cell imbalance, voltage limits, and thermal limits.

The proposed controller combines **Soft Actor-Critic (SAC)** with a **Set Transformer-style feature extractor**. A shared per-cell embedding, multi-head self-attention, learned attention pooling, and a latent projection represent the multi-cell pack before the actor and critic select actions. The thesis evaluates this architecture against a **SAC + MLP** baseline.

**Source fidelity:** The contents of `src/battery_smart_charging/` now derive from the author's uploaded notebook, [`wattention (28).ipynb`](notebooks/original_wattention.ipynb), rather than the earlier illustrative reference implementation. A sanitized copy of the complete original notebook (30 cells) is kept under `notebooks/`. The refactor reorganizes code, adds CLI tools and test seams, and documents limited non-scientific corrections. See [source provenance](docs/source-provenance.md).

## Thesis-reported results

**The following observations come from Chapter 5 of the supplied thesis PDF, not from notebook outputs or runs of this code.** They refer to selected simulation traces and are **not aggregate statistics**.

| Thesis observation | SAC + attention example |
| --- | --- |
| SOC target | **85%** |
| Approximate time until cells converge in SOC | **350–450 s** |
| End of representative charging episode | **~1,100 s**, with SOC around **0.85–0.86** |
| Final terminal voltage range | **~3.97–3.99 V** |
| Final core temperature range | **~28–29 °C** |

The thesis illustrates a comparison to **SAC + MLP**, whose representative simulation exhibits visibly greater cell imbalance, larger charging-command fluctuations, and higher temperatures (some approaching **38–40 °C**). The experimental cases do not establish a percentage improvement with confidence intervals or a cross-seed statistical guarantee. See [thesis results and interpretation](docs/thesis-results.md).

**The README intentionally does not use cell outputs, saved checkpoints, or on-the-fly script results as its source for thesis performance numbers.**

## Implementation details from the uploaded notebook

| Aspect | Value / behavior |
| --- | --- |
| Battery pack | **4 cells** by default; external parameter table `PANmodel.mat` |
| Model | Equivalent-circuit model with SOC, polarization current, hysteresis, and temperature-dependent parameters; two-node lumped thermal model |
| State per cell | SOC, normalized terminal voltage, normalized core temperature (interleaved in the policy observation) |
| Controller action | 4 charging-current commands and 1 coolant command, continuous values in `[0, 1]` |
| Action scaling | Original notebook uses `current_ratio=-100` and `coolant_ratio=2*T` |
| Target / episode limit | `target_soc=0.85`; `MAX_STEP=3000`; `dt=1` s |
| Safety limits | Terminal voltage `2.3–4.2 V`; core temperature `10–40 °C` |
| SAC settings | `gamma=0.95`; `ent_coef=0.01`; buffer `1,000,000`; batch `1024` |
| Attention encoder | 4 heads, embedding width 32, latent features 64, learned cell pooling |
| Actor / critic heads | `[128,64]` hidden units |
| Initial SAC learning rate | `1e-3`; notebook resumed training at `3e-4` |
| Notebook training continuation | `1,700,000` steps in the shown session; **not** the full cumulative training history |

The notebook's cost includes balancing, target SOC, temperature, and voltage terms, adaptive priorities, time pressure, safety shaping, and bounded progress shaping. **The source also computes an action-smoothness diagnostic (`C_smooth`) but does not actually add it to the training reward**. We have not silently introduced a new penalty or claimed the implementation optimizes smoothness explicitly.

## System architecture

```text
         4-cell battery pack (original electro-thermal simulator)
                              |
              per-cell SOC | voltage | temperature
                              |
                  cell-interleaved observation
                              |
              +---------------+------------------+
              |                                  |
         SAC + MLP                         SAC + attention
         baseline                         shared cell embedding
                                          multi-head self-attention
                                          learned weighted pooling
              |                                  |
              +------------------+---------------+
                                 |
                     continuous SAC control
                 4 cell currents + cooling action
                                 |
                safety checks + shaped reward
                                 |
                        pack model update
```

## Project structure

```text
battery-smart-charging-thesis/
├── src/battery_smart_charging/
│   ├── physics.py          # Author's electrical/thermal/pack models (notebook cells 3–6)
│   ├── env.py              # Gymnasium environment and original reward formula (cell 7)
│   ├── features.py         # Author's SetTransformerExtractor (cell 13)
│   ├── callbacks.py        # Best-model/replay-buffer callback (cell 11)
│   ├── visualization.py    # Optional SOC display (cell 0)
│   ├── train.py            # Repeatable SAC training / resume CLI
│   └── evaluate.py         # Deterministic evaluation / CSV export CLI
├── notebooks/
│   └── original_wattention.ipynb  # Full original code, outputs cleared
├── models/
│   ├── PANmodel.mat        # Author-supplied MATLAB battery model (private repository)
│   └── README.md           # File provenance, schema and checksum
├── scripts/               # Smoke check and monitor plotting
├── tests/                 # Synthetic unit tests; no fabricated thesis results
├── docs/                  # Research evidence, provenance, reproducibility
├── .github/workflows/ci.yml
├── pyproject.toml
├── CITATION.cff
└── NOTICE.md
```

## Requirements and setup

The author has now supplied the previously missing **MATLAB battery parameter file** as a separate upload, and it is committed unchanged at [`models/PANmodel.mat`](models/PANmodel.mat) **in this private repository**. The `model` struct contains the temperature-dependent ECM parameters and SOC/OCV tables. Its SHA-256 checksum and provenance are recorded in [`models/README.md`](models/README.md). The loader uses SciPy and no Kaggle mount path is required. Rights to publish the model publicly have **not** been established.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e '.[rl,plot,dev]'
pytest -q
python scripts/smoke_env.py --model-mat models/PANmodel.mat
```

## Train an attention-based SAC policy

```bash
python -m battery_smart_charging.train \
  --model-mat models/PANmodel.mat \
  --policy attention \
  --timesteps 1700000 \
  --log-dir runs/attention \
  --device auto
```

For a **new comparison run**, train the MLP policy separately:

```bash
python -m battery_smart_charging.train \
  --model-mat models/PANmodel.mat \
  --policy mlp \
  --timesteps 1700000 \
  --log-dir runs/mlp
```

The original notebook's initial policy uses `learning_rate=1e-3`, fixed entropy coefficient `0.01`, and SAC with attention. A separate Kaggle session continues training from an existing checkpoint and replay buffer. If you have those artifacts, use `--resume-model`, `--resume-replay-buffer`, and optionally `--learning-rate 3e-4`. Training anew is **not** equivalent to recovering the author's original long training history.

## Evaluate checkpoints

```bash
python -m battery_smart_charging.evaluate \
  --model-mat models/PANmodel.mat \
  --model-path runs/attention/final_model.zip \
  --episodes 5 \
  --output-csv outputs/attention_evaluation.csv
```

The notebook's specific evaluation fixes the normalized cooling action at **0.35**. To intentionally match that choice, add `--cooling-command 0.35`. Otherwise, evaluation leaves cooling under policy control. New evaluation uses the correct **cell-interleaved observation layout**, and exports new experimental data separately from the thesis report.

Monitor plots from your own runs can be generated with:

```bash
python scripts/plot_monitor.py runs/attention/monitor.csv --output outputs/attention_monitor.png
```

## Source integrity and reproducibility limitations

- **Original source:** [`notebooks/original_wattention.ipynb`](notebooks/original_wattention.ipynb), code cells preserved with outputs cleared. Kaggle paths in this archival notebook still refer to the author's original environment.
- **Refactored software:** The physical dynamics, reward equations, state representation and attention extractor are from the uploaded notebook. `env.py` corrects the misleading training-log `success` flag: the original field contained `had_safety_violation`. The new `safety_violation_ever` field retains that diagnostic. The target/termination logic and reward are not rewritten.
- **Original model available:** `models/PANmodel.mat` is the supplied MATLAB battery parameter file; its format and a smoke-run of the physics model are covered by tests. **Still missing:** original checkpoint archives, replay buffers, and monitor/evaluation records. **Scientific reproduction of the reported thesis curves has not been verified.**
- **Baseline:** The MLP training CLI is a comparison convenience, not a recovered MLP-baseline notebook or original baseline checkpoint.
- **Security and physical safety:** This is a simulation. Do not connect control outputs to real cells without independent verification and hardware safety systems.

Further details: [reproducibility](docs/reproducibility.md), [provenance](docs/source-provenance.md), [thesis evidence](docs/thesis-results.md), [rights](NOTICE.md).

## Citation and publication

Cite the **thesis** for published research findings; see [CITATION.cff](CITATION.cff). No public license is assumed, since the manuscript contains a university rights declaration. Confirm publication and distribution permissions before making a public repository.

The code is hosted in the [private GitHub repository](https://github.com/Arian-Abdollahi/lithium-ion-smart-charging-drl). Do not change its visibility until the university's research rights and the model file's distribution permissions have been reviewed. See the [publication guide](docs/publish-to-github.md).
