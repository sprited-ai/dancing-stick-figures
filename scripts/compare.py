"""Compare YOUR checkpoint against the released baselines with the oracle, in one command.

    python scripts/compare.py --ckpt runs/img64/ckpt.pt --cache data/cache [--ref unet_img64] [--n 128]

Downloads the reference checkpoint from sprited/dancing-stick-figures-baselines, scores both (same noise seeds, same
number of samples) plus a real-frame reference, and prints a table. Image checkpoints only (T=1); for video use
eval/run_ckpt.py.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from huggingface_hub import hf_hub_download

REFS = ["unet_img64", "dit_img64_p2", "unet_img64_30k", "dit_img64_p4_30k", "unet_img128", "dit_img128_p4"]


def score(ckpt, cache, n, steps, out):
    subprocess.run([sys.executable, "-m", "eval.score_images", "--ckpt", ckpt, "--cache", cache, "--n", str(n), "--steps", str(steps), "--out", out],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.load(open(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--cache", required=True)
    ap.add_argument("--ref", default="unet_img64", choices=REFS); ap.add_argument("--n", type=int, default=128); ap.add_argument("--steps", type=int, default=50)
    a = ap.parse_args(); os.makedirs("out", exist_ok=True)
    ref_path = hf_hub_download("sprited/dancing-stick-figures-baselines", f"{a.ref}.pt")
    mine = score(a.ckpt, a.cache, a.n, a.steps, "out/compare_mine.json")
    ref = score(ref_path, a.cache, a.n, a.steps, "out/compare_ref.json")
    rows = [("yours  (step %d)" % mine["step"], mine), ("%s (step %d)" % (a.ref, ref["step"]), ref)]
    print(f"\n{'':32s} {'lie↓':>7s} {'tvr↓':>7s} {'cpe↓':>7s} {'clean↑':>7s}")
    for name, m in rows: print(f"{name:32s} {m['lie']:7.3f} {m['tvr']:7.3f} {m['cpe']:7.3f} {m['clean_frac']:7.2f}")
    f = ref["floor"]; print(f"{'real-frame reference':32s} {f['lie']:7.3f} {f['tvr']:7.3f} {f['cpe']:7.3f} {f['clean_frac']:7.2f}")
    print("\nlie = expected colour connections that fail to touch; tvr = missing or fragmented limb colours;")
    print("cpe = impure colours; clean = frames with zero lie/tvr errors.")
    print("Occlusion makes real-reference scores non-zero. Lower is not automatically more realistic: read the metrics together.")


if __name__ == "__main__":
    main()
