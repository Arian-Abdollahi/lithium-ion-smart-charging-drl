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

The private repository now includes the author-uploaded `models/PANmodel.mat`, but still omits trained models, replay buffers, thesis PDF, and original copyrighted figures. The model has no verified public redistribution license. Before changing repository visibility to public, remove the model from the **entire published history** (and from generated release archives), not merely the current working tree. Refer to `models/README.md` for provenance and checksum.

Do not assume the README's thesis-reported numbers have been reproduced by the new modular code until all external model/training assets are available and an equivalence study has been completed.
