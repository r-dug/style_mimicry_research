# Project Plan

This project evaluates perturbation-based anti-style-mimicry protections against modern text-to-image pipelines, with emphasis on Stable Diffusion 3.5 (MM-DiT architecture) and robust fine-tuning countermeasures.

## Research Questions

1. How well does style mimicry perform with newer model architectures (SD3.5), and how much protection remains from existing defenses?
2. Do updated protection tools remain robust against known preprocessing countermeasures (for example, noisy upscaling)?
3. Is robustness meaningfully style-dependent across different artistic styles?

## Canonical Layout and Naming

- Data root: `data/style_mimicry/`
- Protection methods: `glaze`, `mist`, `anti_dreambooth`, `style_guard`
- Countermeasure methods: `diff_pure`, `gaussian_noise`, `noisy_upscaling`, `impress`
- Required minimum samples per artist for training: `20` (no exceptions)

## Step 1: Collect Original Art Samples

- Source directories:
  - `data/style_mimicry/original_art/historical/<style>/<artist>/`
  - `data/style_mimicry/original_art/contemporary/<style>/<artist>/`
- Current execution scope: historical artists first.
- Target: each artist folder must contain at least 20 images.
- Script: `data/style_mimicry/get_historical_art.py`
  - Supports strict target fill behavior.
  - Logs per-artist status through JSON bookmarks.

## Step 2: Apply Protections

- Input: `data/style_mimicry/original_art/`
- Output: `data/style_mimicry/protected_art/<protection>/<split>/...`
- Code location: `mimicry/protections/`

Protections to evaluate:

- GLAZE
- Mist
- Anti-DreamBooth
- StyleGuard

## Step 3: Fine-Tune Stable Diffusion 3.5 (Large)

- Target model baseline: SD3.5 Large.
- Training stacks to trial:
  - SimpleTuner
  - Hugging Face Diffusers `train_dreambooth_lora_sd3.py`
- Trial strategy before full run:
  - One pilot artist per style, then choose primary stack for full experiments.
- Artifact policy:
  - Upload model artifacts automatically to Hugging Face.
  - Repositories private by default.
  - Do not store long-term model weights in this local repo.

Robust preprocessing methods before training:

- `gaussian_noise`
- `diff_pure`
- `noisy_upscaling`
- `impress`

Image preprocessing policy:

- Configurable via flags.
- Defaults:
  - resolution: `(1024, 1024)`
  - aspect handling: `center-crop`
- Also support: `pad`.

## Step 4: Generate Style-Mimic Outputs

- Use a fixed prompt list per experiment condition.
- Output root: `data/style_mimicry/mimic_art/`
- Generated samples should be tagged by artist, protection condition, and training method.

## Step 5: Run Algorithmic Benchmarking

- Use the benchmark framework from DiffAdvPerturbationBench where appropriate.
- Reproducibility pin (current reference): `vkeilo/DiffAdvPerturbationBench@81daa6443552a05230052a1c782af3fb837b5669`
- Benchmark setup details should be documented in `README.md`.

## Step 6: Conduct Human Survey

- Defer full survey protocol finalization until first model outputs are available.
- Human evaluation should assess:
  - style adherence
  - image quality

## Step 7: Analyze Results

- Compare benchmark metrics with human survey outcomes.
- Identify agreement/disagreement patterns between algorithmic and human judgments.
- Analyze style-dependent robustness trends.

## Progress Tracking and Bookmarks

- Source-of-truth tracker: JSON
  - `data/style_mimicry/progress/progress_tracker.json`
- Tracker responsibilities:
  - snapshot current filesystem state
  - step-level progress status
  - per-job bookmarks to skip already-completed work
  - visibility into incomplete artist coverage and pipeline gaps
- Tracking tooling:
  - `mimicry/track_progress.py`
