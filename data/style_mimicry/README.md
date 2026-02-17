# Data Section

This directory is the canonical data root for the project:

`data/style_mimicry`

## Directory Layout

- `original_art/`: source images (by split/style/artist)
- `protected_art/`: Step 2 outputs (by method/split/style/artist)
- `robust_samples/`: countermeasure outputs
- `mimic_art/`: generated mimic outputs
- `models/`: local model cache/downloads
- `progress/progress_tracker.json`: snapshot + bookmark state
- `get_historical_art.py`: Step 1 historical scraper/fill tool

Expected hierarchy examples:

- `original_art/historical/<style>/<artist>/*.jpg`
- `protected_art/mist/historical/<style>/<artist>/*.jpg`

## Step 1: Collect Historical Original Art

Run with explicit data root:

```bash
python3 data/style_mimicry/get_historical_art.py --data-root data/style_mimicry --target-count 20
```

Preview-only:

```bash
python3 data/style_mimicry/get_historical_art.py --data-root data/style_mimicry --target-count 20 --dry-run
```

Scope to a subset:

```bash
python3 data/style_mimicry/get_historical_art.py --data-root data/style_mimicry --style renaissance --limit-artists 3
```

Notes:
- The script fills each artist folder up to `--target-count`.
- Existing files are kept; near-duplicate filenames are cleaned before new downloads.
- Per-artist bookmark status is written into the tracker for resumable runs.

## Progress Tracker

Refresh and print progress summary:

```bash
python3 mimicry/track_progress.py --data-root data/style_mimicry --tracker-path data/style_mimicry/progress/progress_tracker.json --pretty
```

Tracker state includes:
- filesystem snapshots for original/protected/robust/mimic trees
- inferred pipeline step status
- bookmark records for completed/incomplete/failed units of work

## Data Management Rules

- Keep canonical method names in folder paths (`mist`, `anti_dreambooth`, etc.).
- Preserve split/style/artist directory structure across all data products.
- Avoid manual renames/moves inside tracked folders unless intentional and documented.
