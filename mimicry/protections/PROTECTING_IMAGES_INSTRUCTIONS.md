# Step 2: Applying Protections to Original Art

## Overview

This step applies perturbation-based protections to original art images. Each protection method adds adversarial noise designed to disrupt style mimicry fine-tuning while remaining visually imperceptible. Four protection methods are evaluated:

| Method | Source | Approach |
|---|---|---|
| **Mist** | [psyker-team/mist-v2](https://github.com/psyker-team/mist-v2) | PGD adversarial perturbation targeting the UNet |
| **Anti-DreamBooth** | [VinAIResearch/Anti-DreamBooth](https://github.com/VinAIResearch/Anti-DreamBooth) | ASPL (Alternate Surrogate PGD Learning) |
| **StyleGuard** | [PolyLiYJ/StyleGuard](https://github.com/PolyLiYJ/StyleGuard) | PGD with optional style transfer loss |
| **Glaze** | [glaze.cs.uchicago.edu](https://glaze.cs.uchicago.edu/) | Closed-source GUI application (manual setup required) |

## Prerequisites

- Python 3.12 with `diffusers`, `accelerate`, `transformers`, `torch` installed
- GPU with 12GB+ VRAM
- Step 1 complete (original art in `data/style_mimicry/original_art/`)
- Protection tool source code cloned into `mimicry/protections/tools/`

### Fetching Tool Source Code

```bash
bash mimicry/protections/get_protections.sh
```

This clones Mist, Anti-DreamBooth, and StyleGuard into `mimicry/protections/tools/`. Glaze is closed-source and must be set up manually (see Glaze section below).

## File Structure

```
mimicry/protections/
    apply_protections.py            # Orchestrator (runs all methods across all artists)
    apply_protections.sh            # Shell convenience wrapper
    runner_common.py                # Shared runner config/execution helpers
    catalog.py                      # Method metadata and validation
    mist/
        run_protection.py           # Runner shim
        runner_config.json          # Command template
        local_mist_wrapper.py       # Wrapper that invokes mist.py
    anti_dreambooth/
        run_protection.py
        runner_config.json
        local_anti_dreambooth_wrapper.py
    style_guard/
        run_protection.py
        runner_config.json
        local_style_guard_wrapper.py
        precompute_centroids.py     # Pre-compute style centroids for auto-target selection
    glaze/
        run_protection.py
        runner_config.example.json  # Template (copy to runner_config.json after setup)
        setup_glaze_vm.sh           # QEMU/KVM VM setup for running Glaze
    tools/                          # Cloned source repos (not tracked in git)
        mist/
        anti_dreambooth/
        style_guard/
```

## Quick Start

```bash
# 1. Fetch protection tool source code
bash mimicry/protections/get_protections.sh

# 2. Dry-run to verify setup
python mimicry/protections/apply_protections.py --method mist --dry-run

# 3. Run Mist on all historical artists
python mimicry/protections/apply_protections.py --method mist

# 4. Run all available methods
python mimicry/protections/apply_protections.py
```

Or use the shell wrapper with defaults:

```bash
bash mimicry/protections/apply_protections.sh
```

## Usage

### Orchestrator: `apply_protections.py`

```bash
# Run a specific method
python mimicry/protections/apply_protections.py --method mist

# Run multiple methods
python mimicry/protections/apply_protections.py --method mist --method style_guard

# Filter by style or artist
python mimicry/protections/apply_protections.py --method mist --style impressionism
python mimicry/protections/apply_protections.py --method mist --artist Claude_Monet

# Limit number of artists (useful for testing)
python mimicry/protections/apply_protections.py --method mist --limit-artists 2

# Skip already-processed artists (default behavior)
python mimicry/protections/apply_protections.py --method mist

# Force reprocessing
python mimicry/protections/apply_protections.py --method mist --overwrite

# Placeholder mode (copies originals without perturbation, for testing the pipeline)
python mimicry/protections/apply_protections.py --mode placeholder_copy
```

### Execution Modes

| Mode | Description |
|---|---|
| `external_batch_artist` (default) | Calls method wrapper with `--input-dir`/`--output-dir` per artist |
| `external` | Calls method wrapper with `--input`/`--output` per image |
| `placeholder_copy` | Copies originals unmodified (for pipeline testing only) |

## Output Structure

Protected images are saved to:

```
data/style_mimicry/protected_art/{method}/{split}/{style}/{artist}/
```

Example:
```
data/style_mimicry/protected_art/mist/historical/impressionism/Claude_Monet/
data/style_mimicry/protected_art/style_guard/historical/baroque/Caravaggio/
```

## Method-Specific Notes

### Mist

- Uses PGD adversarial perturbation targeting the SD 1.5 UNet
- Runs in single-process mode (`--num_processes 1`)
- Key parameters in `runner_config.json`: `--pgd-alpha 0.005`, `--pgd-eps 0.04`, `--resolution 512`
- Low VRAM mode enabled by default

### Anti-DreamBooth

- Uses ASPL (Alternate Surrogate PGD Learning) to craft perturbations
- Runs in single-process mode
- Key parameters: `--max-train-steps 20`, `--pgd-eps 0.05`, `--resolution 512`

### StyleGuard

- PGD perturbation with optional style transfer loss
- Style loss pushes perturbed images away from the original artist's latent statistics toward a dissimilar target artist
- Auto-target selection enabled by default (`--auto-select-target`)

**Pre-computing style centroids (required for auto-target selection):**

```bash
python mimicry/protections/style_guard/precompute_centroids.py \
    --auto-download-model \
    --model-cache-dir data/style_mimicry/models
```

This encodes all artists through the SD 1.5 VAE and saves channel-wise mean/variance centroids to `data/style_mimicry/style_centroids.json`. The most dissimilar artist (by cosine distance) is automatically selected as the style loss target during protection.

### Glaze

Glaze is a closed-source application distributed as a Windows/Mac GUI. It cannot be run as a CLI command.

**Option 1: QEMU/KVM virtual machine (Linux hosts)**

```bash
cd mimicry/protections/glaze
chmod +x setup_glaze_vm.sh
./setup_glaze_vm.sh          # first-time setup
./setup_glaze_vm.sh start    # start the VM
```

1. Download a Windows evaluation ISO from Microsoft and place it in `glaze/vm/iso/`
2. Connect via VNC: `vncviewer localhost:5900`
3. Install Windows, then install Glaze from https://glaze.cs.uchicago.edu/
4. Transfer images via shared folder: `\\10.0.2.4\qemu` in Windows Explorer
   - Input images: `glaze/vm/shared/input/`
   - Glazed output: `glaze/vm/shared/output/`

**Option 2: Run Glaze natively on a Windows/Mac machine**

Process images through the Glaze GUI and copy the outputs to:
```
data/style_mimicry/protected_art/glaze/{split}/{style}/{artist}/
```

**Note:** Glaze has known floating-point arithmetic issues on some NVIDIA GPUs (e.g., GTX 1660). Use CPU mode in the Glaze settings if you encounter visual artifacts.

After processing, copy `runner_config.example.json` to `runner_config.json` if you want the orchestrator to recognize Glaze as configured.

## Progress Tracking

Protection progress is tracked in `data/style_mimicry/progress/progress_tracker.json` under step `step_2_apply_protections`. Bookmark keys follow the format:

```
mist/historical/impressionism/Claude_Monet
```

The orchestrator automatically skips artists whose output directories already contain all expected images. Use `--overwrite` to force reprocessing.

## Troubleshooting

**Runner preflight fails**: Ensure tool source code exists in `mimicry/protections/tools/`. Run `bash mimicry/protections/get_protections.sh`.

**Out of VRAM**: Mist supports `--low-vram-mode`. Anti-DreamBooth and StyleGuard can reduce `--train-batch-size` to 1.

**Corrupt output images**: If you see "unrecognized data stream" errors, ensure `--num_processes 1` is set in the wrapper (it should be by default). This prevents multi-GPU race conditions when writing output files.

**Non-deterministic outputs**: All `iterdir()` calls are wrapped in `sorted()` to ensure deterministic file ordering. If you still see mismatches, check that input filenames are consistent.
