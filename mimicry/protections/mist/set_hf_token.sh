#!/usr/bin/env bash
set -euo pipefail

TOKEN_PATH="data/style_mimicry/secrets/hf_token.txt"
TOKEN_VALUE="${1:-${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}}"

if [[ -z "${TOKEN_VALUE}" ]]; then
  echo "Usage: bash mimicry/protections/mist/set_hf_token.sh <hf_token>"
  echo "Or export HF_TOKEN/HUGGINGFACE_HUB_TOKEN before running."
  exit 2
fi

mkdir -p "$(dirname "${TOKEN_PATH}")"
printf "%s" "${TOKEN_VALUE}" > "${TOKEN_PATH}"
chmod 600 "${TOKEN_PATH}"
echo "Wrote token to ${TOKEN_PATH}"

