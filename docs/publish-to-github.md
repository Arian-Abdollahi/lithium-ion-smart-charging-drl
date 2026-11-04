# Sync the thesis research repository to GitHub

**Repository:** <https://github.com/Arian-Abdollahi/lithium-ion-smart-charging-drl>

**Privacy:** This repository should remain **private** until university/publication rights have been checked.

This project includes a standard local Git history. If you downloaded the ZIP archive, extract it **including `.git/`** and run from the project directory:

```bash
git remote add origin https://github.com/Arian-Abdollahi/lithium-ion-smart-charging-drl.git
git fetch origin
```

The GitHub version may contain commits published through the GitHub integration rather than the same commit identifiers as the local source history. To synchronize without rewriting either history, compare the two carefully; the simplest fresh local setup is to clone the GitHub version:

```bash
git clone https://github.com/Arian-Abdollahi/lithium-ion-smart-charging-drl.git
cd lithium-ion-smart-charging-drl
python -m pip install -e '.[rl,plot,dev]'
```

The project deliberately omits `PANmodel.mat`, trained models and replay buffers, thesis PDF, and original copyrighted figures. Add the authorized MATLAB model at `models/PANmodel.mat` locally; Git will ignore it.

Do not assume the README's thesis-reported numbers have been reproduced by the new modular code until all external model/training assets are available and an equivalence study has been completed.
