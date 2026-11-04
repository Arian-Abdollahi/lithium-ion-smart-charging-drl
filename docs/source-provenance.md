# Source provenance and changes

## Supplied primary sources

1. **Thesis PDF:** `AUTthesis(4).pdf`, 106-page Persian thesis, especially Chapters 4–6. The README's *performance observations* are transcribed/paraphrased from this PDF; the notebook and code outputs are **not** substituted for the thesis results.
2. **Original notebook:** `wattention (28).ipynb`, 30 cells, supplied by the author. Preserved at [`../notebooks/original_wattention.ipynb`](../notebooks/original_wattention.ipynb) with code text unchanged and execution outputs/ephemeral execution metadata removed.
3. **LaTeX thesis archive:** `Thesis_ArianAbdollahi(2).rar`; manuscript and image assets, not a downloadable MATLAB parameter model or trained policy/replay file.
4. **Author's Kaggle page:** <https://www.kaggle.com/code/arianabdollahi/battery-smart-charging>. A citation and external access point; actual code migration was based on the **uploaded `.ipynb`**, not on an assumption that a live Kaggle page could be inspected.

## Code-to-notebook map

| Module | Notebook source | What changed |
| --- | --- | --- |
| `physics.py` | cells 3, 4, 5, 6 | Moved code into one importable module. Added an optional in-memory model-data input for synthetic unit tests, while normal execution loads the original `.mat` using `mat4py`. |
| `env.py` | cell 7 | Updated local imports; checks that a supplied model file exists; changed the mutable `reset(options={})` default to `None`; calls Gymnasium's base seed initializer. Preserved the transition equations and reward. |
| `features.py` | cell 13 | Original learned embedding, self-attention, learned pooling, and projection. Added a descriptive error in place of the original shape `assert`. |
| `callbacks.py` | cell 11 | Moved original callback to a module; no change to the reward ranking method. |
| `visualization.py` | cell 0 | Separated optional OpenCV-based SOC image generation. |
| `train.py` | cells 14–23 | Original SAC settings and attention class are wired into a parameterized CLI. Training now does not require Kaggle-specific folders and supports optional checkpoint/replay-buffer resume. |
| `evaluate.py` | cells 24–26 | Deterministic evaluation CLI with optional fixed cooling. Correctly handles the notebook's *interleaved* cell observation layout. |
| `scripts/plot_monitor.py` | cells 27–28 | Uses actual SB3 Monitor column names and accepts explicit file paths. It plots newly recorded runs only. |

## Significant fixes and open limitations

- **Misnamed success field:** The uploaded environment had `info['success'] = bool(self.had_safety_violation)`. This reports prior safety **violations**, not charging success. The extracted code now logs `success = is_done and not had_safety_violation`, with the historical meaning exposed under `safety_violation_ever`. This changes **logged metadata**, not reward or state dynamics. Old and new `success` monitor CSV columns are **not** directly interchangeable.
- **Observation layout:** Original environment returns `next_state.T.reshape(-1)` with layout `[SOC_1, V_1, T_1, SOC_2, V_2, T_2, ...]`. The notebook evaluation reshaped it as `(3, n_cells)`, mixing up quantities. `evaluate.py` correctly uses `(n_cells,3)` first.
- **Smoothness in objective:** Original reward computes `C_smooth` but does not multiply it into `cost` or `reward`. The architecture description in the thesis includes smoothness aims, but **this specific notebook code does not directly penalize action changes**. This has deliberately **not** been patched because doing so would change the policy's objective.
- **Cooling in evaluation:** Original evaluation overrides the last action with `0.35`, even though training treats cooling as controlled. This is made optional and explicit.
- **External model:** `PANmodel.mat` is referenced at `/kaggle/input/datasets/arianabdollahi/aaaaaaaa/PANmodel.mat`; the data was **not supplied**, so exact simulation and training cannot yet be reproduced. No substituted calibrated coefficients have been claimed.
- **External training artifacts:** Notebook references pre-existing monitor files, model ZIPs and replay buffers stored in Kaggle datasets. These are not embedded in the uploaded `.ipynb`.
- **Original notebook executable order:** The archival notebook includes notebook magics, hard-coded Kaggle paths, checkpoint archives, and previously created files. It is a provenance copy, **not** a standalone reproducible execution script. The importable package and CLI are the supported interfaces.

## Historical note

This Git repository initially contained a **separately authored, thesis-inspired reference implementation**. The notebook-derived source **replaced** that illustrative simulation when the actual notebook was uploaded. Earlier code is retained in Git history; it is not represented as the author's original experiment.
