# Backlog 04 — Reproducibility Package

## Actions
- Clean `README` with exact setup (uv, SUMO_HOME, libsumo), and a one-command path to reproduce a
  headline result and to regenerate all figures (`figures/make_figures.py --all`).
- Pin the environment (`pyproject.toml` / lockfile). Record SUMO version.
- Commit all Hydra configs and the seed list used for the paper.
- Mint a code DOI (e.g. Zenodo GitHub release) for the submission.
- Run a similarity / plagiarism scan (Turnitin / iThenticate) on the manuscript before submission —
  especially important given reused prose from the FYP.

## Definition of done
A reviewer can clone, install, and reproduce a headline number and all figures from the committed
configs + seeds; a DOI exists; similarity scan passed.
