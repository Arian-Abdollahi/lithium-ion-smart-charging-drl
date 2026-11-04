# Thesis-reported experiments and results

**Source:** *AUTthesis(4).pdf*, Chapter 5 (*Training, Simulation, and Results*) and Chapter 6 (*Conclusions and Suggestions*). The manuscript includes a Persian-language description, reproduced here in English with approximate numerical observations clearly marked.

## Research comparison

The thesis compares a **SAC + MLP** baseline to **SAC + attention**, a set-based representation integrating cell-wise embeddings, multi-head self-attention, and learned pooling. The thesis's reward aims to trade off charge progress, cell imbalance, thermal/voltage constraints, and command variation. It describes a dynamic weighting/gating mechanism that emphasizes balance first and charge completion later.

## Configuration documented by the thesis

| Parameter | Reported value | Location |
| --- | --- | --- |
| Target SOC | `0.85` | Ch. 5: training settings |
| Maximum steps per episode | `3000` | Ch. 5: simulation environment |
| SAC discount | `0.95` | Ch. 5: training settings |
| SAC replay buffer | `1,000,000` | Ch. 5: training settings |
| SAC minibatch | `1024` | Ch. 5: training settings |
| Fixed SAC entropy coefficient | `0.01` | Ch. 5: training settings |
| Attention `d_model`, heads, latent | `32`, `4`, `64` | Ch. 5: Set Transformer feature extractor |
| Actor / critic architecture | `[128,64]` / `[128,64]` | Ch. 5: feature extractor |

## Illustrative episode: SAC + MLP (without attention)

The discussion of Figure `episode_baseline` in Chapter 5 (PDF page 87) describes initial SOC levels of roughly **0.23–0.39**, remaining cell spread of roughly **0.80–0.89** toward the end of the illustrated charge, a cell voltage approaching **4.1 V**, and thermal behavior around **38–40 °C** for some cells. Command profiles show noticeable fluctuations. **These are descriptions of one illustrated trace**, not population statistics.

## Illustrative episode: SAC + attention

The discussion of Figure `episode_attn` in Chapter 5 (PDF pages 88–89) describes:

- Initial SOC spread of roughly **0.26–0.49**, with convergence around **350–450 s**.
- End of the shown charging episode at roughly **1,100 s**, with SOC around **0.85–0.86**.
- Cell terminal voltages ending around **3.97–3.99 V**.
- Temperatures initially approaching **39–40 °C** for some cells, then settling near **28–29 °C** at the end.
- Normalized commands becoming more coherent, around **0.45–0.55** during a stage and **0.25–0.32** toward termination.

The authors interpret this example as consistent with faster balancing, smoother charging actions, and lower final disparity in voltage and temperature compared with the baseline example.

**Comparison caveat:** The initial states and training-curve lengths differ between the illustrated baseline and attention examples. It would be scientifically unsound to infer a precise percentage gain or guarantee that attention is always better from these plots alone. No invented aggregate statistics are presented here.

## Training trends reported

The thesis interprets the attention-based training plots as showing a rising and more stable episodic reward, decreasing episode duration, decreasing `soc_dev` (maximum minus minimum SOC), improvement in minimum SOC, and greater success toward the end of training. The baseline also improves over training, but is described as converging more slowly and with greater fluctuation. Chapter 5 refers to raw training values and moving averages (MA(10); long-window MA(500) for baseline and MA(50) for attention).

## Limits acknowledged in the thesis

Chapter 6 proposes future work on testing under broader initial conditions and sensor noise, battery aging / state-of-health (SOH), automatic entropy tuning, richer attention analysis, practical hardware/slew-rate constraints, and hardware-in-the-loop / real-data validation. These are **future work**, not completed demonstrations.

## Source and reproduction

This summary is based on the **supplied report**, not code-generated metrics. The original thesis PDF and restricted images are intentionally not redistributed here. For source-code limitations, see [source provenance](source-provenance.md).
