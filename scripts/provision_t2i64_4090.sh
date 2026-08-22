#!/usr/bin/env bash
# Provision or smoke-test a fresh RunPod 4090 for the 64px T=1 text-DiT run.
#
# This script deliberately does not create, stop, or terminate a pod, and it
# does not launch the long training run.  The caller supplies an already
# allocated pod's SSH endpoint.
#
#   bash scripts/provision_t2i64_4090.sh HOST PORT provision
#   bash scripts/provision_t2i64_4090.sh HOST PORT smoke
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 POD_HOST POD_PORT provision|smoke" >&2
  exit 2
fi

POD_HOST=$1
POD_PORT=$2
ACTION=$3
REMOTE_ROOT=/workspace/stickdance
SOURCE_HOST=gin
SOURCE_CACHE=/data/dancing-stick-figure-paper/cache/mini
SSH_POD=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p "$POD_PORT" "root@$POD_HOST")

case "$ACTION" in
  provision)
    # The standard RunPod image already contains a suitable CUDA PyTorch.
    # zstd is used only to avoid sending the sparse 5.9 GiB memmap verbatim.
    "${SSH_POD[@]}" "mkdir -p '$REMOTE_ROOT/train' '$REMOTE_ROOT/scripts' '$REMOTE_ROOT/configs' '$REMOTE_ROOT/data/cache64' '$REMOTE_ROOT/runs' && (command -v zstd >/dev/null || (apt-get update -qq && apt-get install -y -qq zstd))"

    tar -cf - \
      train/__init__.py \
      train/video_ddpm.py \
      train/video_dit_fm.py \
      train/requirements.txt \
      scripts/dataset_preflight.py \
      configs/dataset_fingerprints.json \
      | "${SSH_POD[@]}" "tar -C '$REMOTE_ROOT' -xf -"

    # Stream through this workstation; no 6 GiB temporary archive is created.
    # At zstd -1 the observed transfer is about 346 MiB.
    ssh -o BatchMode=yes "$SOURCE_HOST" \
      "tar -C '$SOURCE_CACHE' -cf - frames.npy clips.json meta.json | zstd -1 -q -c" \
      | "${SSH_POD[@]}" "zstd -d -q -c | tar -C '$REMOTE_ROOT/data/cache64' -xf -"

    "${SSH_POD[@]}" "cd '$REMOTE_ROOT' && python -m pip install -q -r train/requirements.txt && \
      python scripts/dataset_preflight.py --cache data/cache64 --profile colored_k1_v1 \
        --out data/cache64/preflight.json --grid data/cache64/reference_grid.png && \
      python - <<'PY'
import torch
assert torch.cuda.is_available()
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(0))
PY"
    ;;

  smoke)
    "${SSH_POD[@]}" "cd '$REMOTE_ROOT' && python scripts/dataset_preflight.py \
      --cache data/cache64 --profile colored_k1_v1 \
      --out data/cache64/preflight-smoke.json --grid data/cache64/reference_grid.png"
    # Batch 128 is intentional: this checks the intended full-run capacity,
    # text encoder, one-frame data path, EMA checkpoint, and sample export.
    "${SSH_POD[@]}" "cd '$REMOTE_ROOT' && python -u -m train.video_dit_fm \
      --cache data/cache64 \
      --out runs/t2i64_text_capacity_smoke \
      --size 64 --frames 1 --stride 1 \
      --patch 4 --dim 384 --depth 12 --heads 6 \
      --cond text --cfg_drop 0.1 \
      --text_encoder google-t5/t5-small --text_len 32 \
      --batch 128 --steps 2 --sample_every 2 --val_every 500 \
      --workers 8 --fast --shift 1.0 --lr_final 1.0 --seed 0"
    "${SSH_POD[@]}" "test -s '$REMOTE_ROOT/runs/t2i64_text_capacity_smoke/ckpt_000002.pt' && test -s '$REMOTE_ROOT/runs/t2i64_text_capacity_smoke/sample_000002.png' && ls -lh '$REMOTE_ROOT/runs/t2i64_text_capacity_smoke/'"
    ;;

  *)
    echo "action must be provision or smoke" >&2
    exit 2
    ;;
esac
