# Reproducibility and research limitations

## Source available

- The exact code for the original notebook's battery-pack model, environment reward/termination, and Set Transformer extractor has been reorganized into package modules.
- A full **output-stripped original notebook** is retained for code comparison.
- The thesis-reported numeric observations in the README were sourced from the PDF, not recomputed from training traces.
- Unit tests use **synthetic parameter values solely for software correctness checks**, never thesis performance claims.

## Required files not supplied

1. **Battery parameter model:** `PANmodel.mat` from the original Kaggle input dataset. It defines temperature-dependent ECM parameter tables, OCV-SOC relationships, and related values. Place an authorized copy at `models/PANmodel.mat` and pass the path explicitly.
2. **Original checkpoints and replay buffers:** The notebook resumes from an existing pre-trained SAC policy and replay buffer. Without these, no faithful reproduction of the continuation session is possible.
3. **Recorded training monitor CSVs and evaluation records:** The original paths in notebook cells 14 and 27–28 resolve to Kaggle datasets absent from this export.
4. **Consistent evaluation settings:** The original cell 26 forces the cooling action to `0.35`. The CLI makes this override explicit and optional.

## Settings recovered from notebook source

- Four cells; 25 °C initial/ambient temperature, 1-second time step.
- Continuous box actions in `[0,1]^5`; current multiplier `-100` and coolant multiplier `2*T`.
- Random initial cell SOC uniform in `[0.2,0.5]` during reset, with cell parameter perturbations.
- Goal: mean SOC near 0.85, 3000 maximum simulation steps.
- Safety: terminal voltages 2.3–4.2 V, temperature 10–40 °C.
- SAC discount 0.95, replay capacity 1,000,000, minibatch 1024, fixed entropy coefficient 0.01.
- Attention: `d_model=32`, 4 heads, latent dimension 64, MLP policy heads 128/64.
- Initial learning rate `1e-3`; resumed learning rate `3e-4`.
- Training cell displays a continuation of `1,700,000` timesteps. It does not prove that this is the full training history.

## Minimal verification commands

```bash
python -m pip install -e '.[rl,plot,dev]'
pytest -q
python scripts/smoke_env.py --model-mat models/PANmodel.mat
python -m battery_smart_charging.train --model-mat models/PANmodel.mat --timesteps 10000
python -m battery_smart_charging.evaluate --model-mat models/PANmodel.mat --model-path runs/attention/final_model.zip --episodes 3
```

The example 10,000 steps are just a small **new run**, not a reproduction of the thesis experiment.

## Scope and limitations

- No end-to-end reproduction of the PDF plots is claimed until exact model parameters, original trained checkpoints, and training data are available.
- The notebook's `info['success']` flag was incorrectly tied to a historical safety violation. This refactor logs a corrected value and exposes the original meaning separately. Comparisons with historical monitor CSVs must account for this.
- The source calculates smoothness cost but does not apply it to reward, despite the prose goals. A future controlled experiment may add it; that would be a **different objective**.
- The pack action model assumes independently actuated cell charging channels. It cannot be deployed on arbitrary series-connected hardware without an appropriate electrical topology and verified hardware safety limits.
- Hardware safety, aging, noisy sensors, current ramp limits and real-cell parameter validation remain outside this repository's verified scope.
