# Style Mimicry Robustness Project

This repository evaluates style-mimicry robustness across perturbation-based protections, robust preprocessing countermeasures, and SD3.5 fine-tuning workflows.

## Canonical Paths

- Data root: `data/style_mimicry`
- Progress tracker: `data/style_mimicry/progress/progress_tracker.json`
- Plan: `Project_PLAN.md`

## Method Naming (Canonical)

- Protections: `glaze`, `mist`, `anti_dreambooth`, `style_guard`
- Countermeasures: `diff_pure`, `gaussian_noise`, `noisy_upscaling`, `impress`

## Reproducibility

- Benchmark framework reference pin:
  - `vkeilo/DiffAdvPerturbationBench@81daa6443552a05230052a1c782af3fb837b5669`
- If you need to refresh this pin later:
  - `git ls-remote https://github.com/vkeilo/DiffAdvPerturbationBench HEAD`

## Progress Tracking

Refresh project state and step coverage:

```bash
python3 mimicry/track_progress.py --pretty
```

The tracker stores:

- file-system snapshots per step
- inferred step-level status
- per-job bookmarks for completed/incomplete/failure outcomes

## Historical Data Collection

Fill historical artists to the strict target count (default 20):

```bash
python3 data/style_mimicry/get_historical_art.py --target-count 20
```

Preview without downloading:

```bash
python3 data/style_mimicry/get_historical_art.py --dry-run
```

## Step 2 Protections

Prepare protection source manifest:

```bash
bash mimicry/protections/get_protections.sh
```

Optionally fetch public repos:

```bash
bash mimicry/protections/get_protections.sh --fetch
```

Install Mist runtime dependencies in your active environment before Mist runs:

```bash
python3 -m pip install -r mimicry/protections/tools/mist/requirements.txt
```

Mist model source handling:

- If `--pretrained-model-name-or-path` is a local existing path, it is used directly.
- If it is a Hugging Face model id (for example `stable-diffusion/stable-diffusion-1-5`), the wrapper first checks local cache.
- If missing and `--auto-download-model` is enabled, it downloads to `--model-cache-dir` (default in config: `data/style_mimicry/models`).
- Hugging Face auth defaults:
  - `--hf-token` if explicitly provided
  - otherwise standard Hugging Face resolution (`HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` / cached `hf auth login`)

### Hugging Face Login (Recommended)

To avoid repeated auth failures and enable faster model downloads, log in once in your environment.

1. Create a User Access Token on Hugging Face:
   - https://huggingface.co/settings/tokens
2. Log in from terminal (recommended command in current Hugging Face CLI):

```bash
hf auth login
```

3. Verify login:

```bash
hf auth whoami
```

4. Optional non-interactive shell setup (useful on servers):

```bash
export HF_TOKEN=your_token_here
```

Notes:
- `HF_TOKEN` is picked up automatically by this project.
- On older CLI versions, the equivalent command may be `huggingface-cli login`.

Source references:
- Hugging Face Hub CLI auth docs: https://huggingface.co/docs/huggingface_hub/en/guides/cli
- `hf auth` command reference: https://huggingface.co/docs/huggingface_hub/en/package_reference/cli
- Python authentication reference (`huggingface_hub.login`): https://huggingface.co/docs/huggingface_hub/en/package_reference/authentication

Fast download defaults:

- `--hf-enable-transfer` (sets `HF_HUB_ENABLE_HF_TRANSFER=1`)
- `--hf-max-workers 16`

Apply protections (per-image mode; real mode expects method runners under each protection folder):

```bash
bash mimicry/protections/apply_protections.sh --mode external --split historical
```

Apply protections in batch-by-artist mode (recommended for Mist):

```bash
bash mimicry/protections/apply_protections.sh --mode external_batch_artist --split historical --method mist
```

No-arg default for `apply_protections.sh`:

```bash
bash mimicry/protections/apply_protections.sh
```

This defaults to `--mode external_batch_artist --method mist --split historical`.

Before `--mode external`, create runner configs for each method:

- `mimicry/protections/glaze/runner_config.json`
- `mimicry/protections/mist/runner_config.json`
- `mimicry/protections/anti_dreambooth/runner_config.json`
- `mimicry/protections/style_guard/runner_config.json`

Use each `runner_config.example.json` as the starting template.
For Mist, a default `runner_config.json` is already provided for batch mode.

Pipeline test mode (copy-only placeholder):

```bash
bash mimicry/protections/apply_protections.sh --mode placeholder_copy --split historical --limit-artists 1
```
