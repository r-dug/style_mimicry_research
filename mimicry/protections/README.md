# Protections (Step 2)

This section applies perturbation-based protection methods to images in `data/style_mimicry/original_art`, writing outputs to `data/style_mimicry/protected_art`.

Canonical methods:
- `glaze`
- `mist`
- `anti_dreambooth`
- `style_guard`

## Folder Layout

- `mimicry/protections/get_protections.py`: source bootstrap/fetch manifest builder
- `mimicry/protections/apply_protections.py`: Step 2 orchestrator
- `mimicry/protections/runner_common.py`: shared runner/config loader
- `mimicry/protections/<method>/run_protection.py`: method-specific launcher
- `mimicry/protections/<method>/runner_config.json`: command template used by launcher
- `mimicry/protections/tools/`: external method source checkouts

## Prepare Method Sources

Create/update the protection source manifest:

```bash
bash mimicry/protections/get_protections.sh
```

Fetch available public repos:

```bash
bash mimicry/protections/get_protections.sh --fetch
```

Notes:
- `mist`, `anti_dreambooth`, and `style_guard` can be fetched from their configured repos.
- `glaze` is marked manual-source-required (no public repo clone path in this project). It can be used via downloadable Windows or MacOS executables.

## Configure Method Runners

Each method launcher reads `runner_config.json` and runs `command_template`.

Supported template placeholders:
- file mode: `{input}` and `{output}`
- artist-folder mode: `{input_dir}` and `{output_dir}`

Validate a runner without processing files:

```bash
python3 mimicry/protections/mist/run_protection.py --check-only
```

## Run Step 2

Project default (no args) uses Mist in artist-folder batch mode on historical split:

```bash
bash mimicry/protections/apply_protections.sh
```

Equivalent explicit command:

```bash
bash mimicry/protections/apply_protections.sh --mode external_batch_artist --method mist --split historical
```

Useful options:
- `--dry-run`: print actions without writing outputs
- `--overwrite`: recompute existing outputs
- `--style <style_name>` / `--artist <artist_name>`: scope processing
- `--limit-artists N`: run a small trial first

## Mist Runtime Notes

Install Mist dependencies in your active environment:

```bash
python3 -m pip install -r mimicry/protections/tools/mist/requirements.txt
```

Authentication/model download behavior:
- uses Hugging Face defaults (`HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` / cached `hf auth login`)
    - `hf auth login --token <your token here>` is the method I use.
- model cache defaults to `data/style_mimicry/models`
- transfer acceleration defaults on (`HF_HUB_ENABLE_HF_TRANSFER=1`, workers=16)

## Outputs and Progress

Step 2 outputs are written under:

`data/style_mimicry/protected_art/<method>/<split>/<style>/<artist>/`

Bookmark/progress state is tracked in:

`data/style_mimicry/progress/progress_tracker.json`
