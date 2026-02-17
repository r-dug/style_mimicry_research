# Project Structure Outline

Reference: `Project_PLAN.md`  
Snapshot date: 2026-02-16

## Top Level

- `Project_PLAN.md`: project goals, constraints, and 7-step pipeline.
- `PROJECT_OPEN_QUESTIONS.md`: decision log for planning assumptions.
- `data/`: dataset and generated experiment artifacts.
- `mimicry/`: pipeline code (protections, countermeasures, tuning, benchmarking, tracking).
- `visualization/`: output figures/plots and reporting assets.
- `background_sources/`: supporting papers/notes.
- `environment.yml`: base environment spec.

## Data Layout (`data/style_mimicry`)

- `original_art/historical/{abstract,baroque,cubism,impressionism,realism,surrealism}/<artist>/`
- `original_art/contemporary/{pop,anime,geometric,photorealism,pop_surrealism}/<artist>/`
- `protected_art/{glaze,mist,anti_dreambooth,style_guard}/{historical,contemporary}/...`
- `robust_samples/{diff_pure,gaussian_noise,noisy_upscaling,impress}/{historical,contemporary}/...`
- `mimic_art/`
- `progress/progress_tracker.json` (JSON source-of-truth for step status and bookmarks)

Current data snapshot:

- Historical artist folders present: 26
- Contemporary artist folders present: 0
- Historical sample images present: 499 total

## Code Layout (`mimicry`)

- `common/constants.py`: canonical method names, image extensions, and step IDs.
- `common/state_getters.py`: getter functions that inspect filesystem state by step.
- `common/progress_tracker.py`: JSON tracker + bookmark helpers.
- `track_progress.py`: CLI to refresh tracker snapshots from current filesystem state.
- `protections/{glaze,mist,anti_dreambooth,style_guard}/`
- `countermeasures/{gaussian_noise,diff_pure,noisy_upscaling,impress}/`
- `tuning/`
- `benchmarking/`

## Pipeline Mapping

- Step 1 (collect originals): `data/style_mimicry/get_historical_art.py` -> `data/style_mimicry/original_art/`
- Step 2 (apply protections): `mimicry/protections/` -> `data/style_mimicry/protected_art/`
- Step 3 (countermeasures + tuning): `mimicry/countermeasures/`, `mimicry/tuning/` -> `data/style_mimicry/robust_samples/`
- Step 4 (mimic generation): outputs under `data/style_mimicry/mimic_art/`
- Step 5 (benchmarking): `mimicry/benchmarking/`
- Step 6 (survey): protocol + collected ratings artifacts (path to be finalized with first outputs)
- Step 7 (analysis): consolidated reporting and visualization
