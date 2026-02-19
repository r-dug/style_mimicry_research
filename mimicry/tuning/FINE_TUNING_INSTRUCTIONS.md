# Step 3: Fine-Tuning SD3.5 Large with DreamBooth LoRA

## Overview

This step trains one LoRA adapter per artist on Stable Diffusion 3.5 Large using HuggingFace Diffusers' DreamBooth LoRA script. Each adapter learns to reproduce an artist's style via a rare token identifier (`sks`).

## Prerequisites

- Python 3.12 with `diffusers`, `accelerate`, `transformers`, `peft`, `torch` installed
- HuggingFace account with gated access to `stabilityai/stable-diffusion-3.5-large`
- `HF_TOKEN` environment variable set (for model download and Hub upload)
- GPU with 24GB+ VRAM (A100, RTX 4090, etc.)
- Step 1 complete (original art collected in `data/style_mimicry/original_art/`)

## File Structure

```
mimicry/tuning/
    apply_tuning.py                          # Orchestrator (runs all artists)
    hf_diffusers/
        local_dreambooth_lora_sd3_wrapper.py # Wrapper (runs one artist)
        train_dreambooth_lora_sd3.py         # Vendored HF diffusers script (do not modify)
        runner_config.json                   # Command template for runner_common
        run_tuning.py                        # Thin runner shim
```

## Quick Start

```bash
# 1. Verify targets with dry-run
python mimicry/tuning/apply_tuning.py --dry-run

# 2. Pilot run (one artist per style)
python mimicry/tuning/apply_tuning.py --pilot

# 3. Full run on all artists
python mimicry/tuning/apply_tuning.py
```

## Usage

### Orchestrator: `apply_tuning.py`

The orchestrator walks the data tree, discovers artist targets, and invokes the wrapper for each.

```bash
# Train on original art (default — no protection, no countermeasure)
python mimicry/tuning/apply_tuning.py

# Train on mist-protected images
python mimicry/tuning/apply_tuning.py --protection mist

# Train on countermeasure-processed images
python mimicry/tuning/apply_tuning.py --protection mist --countermeasure noisy_upscaling

# Pilot mode (one artist per style for trial runs)
python mimicry/tuning/apply_tuning.py --pilot

# Single artist
python mimicry/tuning/apply_tuning.py --artist Claude_Monet

# Disable Hub upload
python mimicry/tuning/apply_tuning.py --no-push-to-hub

# Filter by style
python mimicry/tuning/apply_tuning.py --style impressionism --style baroque

# Override hyperparameters
python mimicry/tuning/apply_tuning.py --max-train-steps 1000 --rank 32 --learning-rate 5e-5
```

### Wrapper: `local_dreambooth_lora_sd3_wrapper.py`

For training a single artist directly (the orchestrator calls this under the hood):

```bash
python mimicry/tuning/hf_diffusers/local_dreambooth_lora_sd3_wrapper.py \
    --input-dir data/style_mimicry/original_art/historical/impressionism/Claude_Monet \
    --output-dir data/style_mimicry/models/lora/hf_diffusers/none__none/historical/impressionism/Claude_Monet \
    --no-push-to-hub
```

## Input Data Resolution

The `--protection` and `--countermeasure` flags control which image set is used for training:

| `--protection` | `--countermeasure` | Source directory |
|---|---|---|
| `none` (default) | `none` (default) | `original_art/{split}/{style}/{artist}/` |
| `mist` | `none` | `protected_art/mist/{split}/{style}/{artist}/` |
| `mist` | `noisy_upscaling` | `robust_samples/noisy_upscaling/{split}/{style}/{artist}/` |

## Output Structure

LoRA weights are saved to:

```
data/style_mimicry/models/lora/{stack}/{protection}__{countermeasure}/{split}/{style}/{artist}/
    pytorch_lora_weights.safetensors
    adapter_config.json
```

Example:
```
data/style_mimicry/models/lora/hf_diffusers/none__none/historical/impressionism/Claude_Monet/
```

## Default Hyperparameters

| Parameter | Default | Rationale |
|---|---|---|
| `--resolution` | 1024 | SD3.5 Large native resolution |
| `--rank` | 16 | Good balance for ~20 training images |
| `--max-train-steps` | 500 | ~25 epochs with effective batch size 4 |
| `--learning-rate` | 1e-4 | Standard for LoRA fine-tuning |
| `--train-batch-size` | 1 | Fits in 24GB VRAM with gradient checkpointing |
| `--gradient-accumulation-steps` | 4 | Effective batch size of 4 |
| `--mixed-precision` | bf16 | Avoids fp16 overflow in DiT attention layers |
| `--gradient-checkpointing` | enabled | Essential for 24GB VRAM |
| `--cache-latents` | enabled | Reduces VRAM by pre-encoding images |
| `--seed` | 42 | Reproducibility |
| `--instance-prompt` | `"a painting in the style of sks"` | DreamBooth rare token binding |

## Progress Tracking

Training progress is tracked in `data/style_mimicry/progress/progress_tracker.json` under step `step_3_fine_tune_sd35`. Bookmark keys follow the format:

```
hf_diffusers/none/none/historical/impressionism/Claude_Monet
```

Already-completed artists (those with `pytorch_lora_weights.safetensors` in their output directory) are automatically skipped on re-runs. Use `--overwrite` to force retraining.

## HuggingFace Hub Upload

By default, trained LoRA adapters are pushed to HuggingFace Hub as private repos. The repo name is auto-generated as:

```
sd35-lora-{protection}-{countermeasure}-{artist}
```

Set `HF_TOKEN` in your environment. Use `--no-push-to-hub` to disable.

## Troubleshooting

**Out of VRAM**: Reduce `--resolution` to 768 or 512. Ensure `--gradient-checkpointing` and `--cache-latents` are enabled (they are by default).

**Model access denied**: Accept the license at https://huggingface.co/stabilityai/stable-diffusion-3.5-large and ensure `HF_TOKEN` is set.

**Training too slow**: Ensure you're using a GPU (`nvidia-smi` to verify). BF16 requires Ampere or newer (A100, RTX 30xx/40xx).

**Hub upload fails**: Check `HF_TOKEN` permissions (needs write access). Use `--no-push-to-hub` to skip and upload manually later.
