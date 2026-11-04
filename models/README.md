# Original battery parameter data

`PANmodel.mat` is the author-supplied MATLAB v5 battery-model file from the
Kaggle experiment (uploaded as `PANmodel (2).mat`). It is tracked **in this
private research repository** as `models/PANmodel.mat` so that a fresh private
clone can run the electrical/thermal simulation without an extra download.

- MATLAB top-level variable: `model` (struct).
- Dataset identifier inside the model: `FORD`.
- Eight temperatures: from −25 to 45 °C.
- Includes SOC/OCV lookup tables, coulombic efficiency, capacity, hysteresis,
  and temperature-dependent equivalent-circuit parameters.
- Size: 11,799 bytes.
- SHA-256: `b90711c02e33bd7a2afbec57202e51f09f7d4ecddaaa2ffe2094736a81b39c2b`.

The code reads this MATLAB v5 file directly with `scipy.io.loadmat`; no
`mat4py` package or Kaggle mount path is needed.

**Rights:** This is an author-uploaded research input, not an assertion of a
public-domain or redistributable dataset. Keep the repository private until
the right to distribute this model publicly has been established. If creating
a public repository, remove the data file from the published history as well,
not just from the latest commit.

Check its integrity and the loader without needing a trained RL policy:

```bash
python -m pytest tests/test_panmodel.py -q
```

Trained checkpoints, replay buffers and historical monitor files are still not
available; possession of the model is not proof the reported thesis results
have been independently reproduced.
