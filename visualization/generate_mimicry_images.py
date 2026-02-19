#!/usr/bin/env python3
"""Generate style mimicry test images for pilot artists using trained LoRA weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusion3Pipeline

LORA_BASE = Path("data/style_mimicry/models/lora/hf_diffusers")
OUTPUT_DIR = Path("visualization")

PILOT = {
    "cubism":       "Fernand_Leger",
    "impressionism": "Berthe_Morisot",
    "realism":      "Gustave_Courbet",
}

PROMPTS = {
    "portrait":  "a portrait of a young woman in the style of sks",
    "landscape": "a city landscape in the style of sks",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LoRA style mimicry test images.")
    parser.add_argument("--protection", default="none")
    parser.add_argument("--countermeasure", default="none")
    parser.add_argument("--split", default="historical")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--style", action="append", default=[],
        help="Limit to specific styles (repeatable). Default: all pilot styles.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    variant = f"{args.protection}__{args.countermeasure}"
    lora_root = LORA_BASE / variant / args.split

    styles = args.style if args.style else list(PILOT.keys())

    print("Loading base SD3.5 pipeline...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    for style in styles:
        artist = PILOT.get(style)
        if artist is None:
            print(f"  skip: no pilot artist for style '{style}'")
            continue

        lora_dir = lora_root / style / artist
        if not lora_dir.exists():
            print(f"  skip {style}/{artist}: no LoRA at {lora_dir}")
            continue

        print(f"\n[{style}/{artist}]")
        try:
            pipe.load_lora_weights(str(lora_dir))

            for label, prompt in PROMPTS.items():
                out_path = args.output_dir / f"{artist}_{label}_mimicry.png"
                print(f"  generating {label}...")
                image = pipe(
                    prompt,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                ).images[0]
                image.save(out_path)
                print(f"  saved: {out_path}")

        except Exception as exc:
            print(f"  ERROR: {exc}")
        finally:
            pipe.unload_lora_weights()

    print("\nDone.")


if __name__ == "__main__":
    main()
