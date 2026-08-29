#!/usr/bin/env python3
"""Evidence-backed numeric audit for the arXiv v6 manuscript."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "dataset_paper_v6.tex"
RESULTS = ROOT / "paper" / "results"
REPORT = ROOT / "paper" / "ARXIV_V6_NUMBER_AUDIT.md"


def load(name: str):
    return json.loads((RESULTS / name).read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def r3(value: float) -> str:
    return f"{value:.3f}".lstrip("0")


def r1(value: float) -> str:
    return f"{value:.1f}"


checks: list[str] = []


def check(condition: bool, description: str):
    if not condition:
        raise AssertionError(description)
    checks.append(description)


tex = PAPER.read_text()
check(not re.search(r"\b10k\b|10[,{]000", tex), "No superseded 10k model row remains")
check("Dancing Stick Figures: An Introductory Dataset for Training Video Generation Models" in tex,
      "Final title is present")

character = load("dataset_characterization_v02.json")
for token in ("134", "1,340", "4,020", "482,400"):
    check(token in tex, f"Dataset count token {token} is present")

palette = load("palette_mask_validation_v02.json")
p60 = palette["thresholds"]["60"]
check(palette["n_frames"] == 4020, "Palette audit uses one frame for each of 4,020 clips")
check(r1(100 * p60["assigned_fraction"]) in tex, "Palette assigned fraction matches evidence")
check(r1(100 * p60["accuracy_given_assigned"]) in tex, "Palette assigned-pixel accuracy matches evidence")
check(r1(100 * p60["macro_iou"]) in tex, "Palette macro IoU matches evidence")

corrupt = load("corruption_500.json")
check(corrupt["n"] == 500, "Controlled structural study has n=500")
for value in (
    corrupt["conditions"]["real"]["lie"]["mean"],
    corrupt["conditions"]["swap_LR_partial"]["lie"]["mean"],
    corrupt["conditions"]["real"]["tvr"]["mean"],
    corrupt["conditions"]["extra_arm"]["tvr"]["mean"],
):
    check(r3(value) in tex, f"Structural-corruption value {r3(value)} matches evidence")

deg = load("degenerate_64f_n128.json")
real_b = deg["baselines"]["real_reference_b"]
check(deg["n"] == 128 and deg["frames"] == 64 and deg["stride"] == 1,
      "64-frame diagnostic control protocol is n=128, stride 1")
for key, token in (("centroid_speed", ".373"), ("centroid_accel", ".415"),
                   ("motion_fraction", ".501"), ("angle_jerk", ".073")):
    check(r3(real_b[key]["mean"]) == token and token in tex,
          f"Real-reference {key} {token} matches evidence")

fvd = load("fvd_64f_n128.json")
for key, token in (("real_reference_b", "114.7"), ("repeat_first", "620.8"),
                   ("shuffle_frames", "520.7"), ("reverse_time", "120.1"),
                   ("loop_first_8", "418.2"), ("train_replay", "125.5")):
    check(r1(fvd["fvd"][key]) == token and token in tex, f"FVD {key} {token} matches evidence")

reverse = load("fvd_64f_reverse_uncertainty.json")
check(reverse["repeats"] == 30 and "30 paired resamples" in tex, "Reverse uncertainty uses 30 resamples")
for value, token in ((reverse["paired_delta"]["mean"], "5.24"),
                     (reverse["paired_delta"]["std"], "5.24"),
                     (reverse["paired_delta"]["range"][0], "-5.76"),
                     (reverse["paired_delta"]["range"][1], "15.74")):
    check(f"{value:.2f}" == token and token in tex, f"Reverse uncertainty {token} matches evidence")
check(round(reverse["paired_delta"]["positive_fraction"] * 30) == 25 and "25/30" in tex,
      "Reverse positive-resample count is 25/30")

loc = load("part_motion_localization_v02_n24.json")
check("24 unflagged test motions" in tex, "Localization study n=24 is stated")
for severity, token in (("0.25", "1.51"), ("0.5", "4.69"), ("0.75", "8.61"), ("1.0", "9.11")):
    value = loc["by_severity"][severity]["target_path_drop"]["median"]
    check(f"{value:.2f}" == token and token in tex, f"Localization severity {severity} value {token} matches")

prompt_fvd = load("fvd_prompt_seed_validation_64f_v02.json")
for branch, tokens in (
    ("same_prompt_roster_different_seeds", ("26.37", "23.72", "30.13")),
    ("random_video_halves", ("24.61", "21.39", "28.03")),
    ("different_prompt_rosters_group_balanced", ("47.82", "39.51", "57.32")),
):
    node = prompt_fvd[branch]
    values = (node["median"], node["range"][0], node["range"][1])
    for value, token in zip(values, tokens):
        check(f"{value:.2f}" == token and token in tex, f"Prompt-FVD value {token} matches evidence")

i3d = load("i3d_embedding_validation_64f_v02c.json")
for value, token in (
    (i3d["clean_pair_distance"]["same_prompt_different_seed"]["median"], "24.21"),
    (i3d["clean_pair_distance"]["different_prompt_same_group"]["median"], "32.82"),
    (i3d["paired_clean_to_corruption_distance"]["repeat_first"]["median"], "29.09"),
    (i3d["paired_clean_to_corruption_distance"]["shuffle_frames"]["median"], "27.52"),
    (i3d["paired_clean_to_corruption_distance"]["loop_first_8"]["median"], "28.46"),
    (i3d["paired_clean_to_corruption_distance"]["reverse_time"]["median"], "9.34"),
):
    check(f"{value:.2f}" == token and token in tex, f"I3D feature value {token} matches evidence")
check(r1(100 * i3d["clean_pair_distance"]["probability_same_prompt_is_closer"]) == "80.4" and "80.4" in tex,
      "I3D same-prompt comparison rate 80.4% matches")

image = load("image_dit_30k_n512.json")
check((image["step"], image["n"], image["steps"], image["unique_prompts"]) == (30000, 512, 50, 134),
      "Single-frame Image DiT was rerun at step 30k, n=512, 50 steps, 134 prompts")
for token in (r3(image["floor"]["tvr"]), r3(image["floor"]["lie"]), r3(image["floor"]["cpe"]),
              r3(image["tvr"]), r3(image["lie"]), r3(image["cpe"])):
    check(token in tex, f"Single-frame table value {token} matches fresh rerun")

codec = load("codec_floor_f4t4d8_64f_n128.json")
video_rows = [
    ("codec", codec["baselines"]["codec_recon"], codec["baselines"]["codec_recon"]["fvd"]),
    ("factorized image", load("pixel_factorised_image30k_win64_n128.json"), None),
    ("factorized random", load("pixel_factorised_random30k_win64_n128.json"), None),
    ("local mixer", load("pixel_localmixer_image30k_win64_n128.json"), None),
    ("Mini-Wan", load("wanmini40_decode_30k_win64_n128.json"), None),
]
for name, row, explicit_fvd in video_rows:
    if name != "codec":
        check((row["step"], row["n"], row["sampling_seeds"], row["target_frames"],
               row["reference_stride"], row["sample_steps"], row["cfg"]) ==
              (30000, 128, [0, 1, 2], 64, 1, 50, 3.0), f"{name} canonical protocol metadata matches")
    values = [row["tvr"] if name != "codec" else row["tvr"]["mean"],
              row["lie"] if name != "codec" else row["lie"]["mean"],
              row["cpe"] if name != "codec" else row["cpe"]["mean"],
              row["centroid_speed"] if name != "codec" else row["centroid_speed"]["mean"],
              row["motion_fraction"] if name != "codec" else row["motion_fraction"]["mean"],
              row["angle_jerk"] if name != "codec" else row["angle_jerk"]["mean"]]
    fvd_value = explicit_fvd if explicit_fvd is not None else row["fvd"]
    for token in [*(r3(v) for v in values), r1(fvd_value)]:
        check(token in tex, f"{name} table value {token} matches evidence")

vae = load("vae_f4_memorization.json")
check(r1(vae["memorization_gap_pct"]) == "9.5" and "9.5" in tex, "Codec memorization gap 9.5% matches")
check(r1(vae["smuggle_damage_x"]) == "5.5" and "5.5" in tex, "Codec background-neutralization factor 5.5x matches")

rebuild = load("rebuild_full_frames_v2.json")
check(rebuild["passed"] and rebuild["matched_rows"] == 514800, "Full 128px rebuild audit passed all 514,800 rows")
for difference, token in (
    (rebuild["color_channel_difference_fraction"], "99.88"),
    (rebuild["seg_pixel_difference_fraction"], "99.92"),
    (rebuild["joint_visibility_bit_difference_fraction"], "99.50"),
):
    check(f"{100 * (1 - difference):.2f}" == token and token in tex, f"Rebuild agreement {token}% matches evidence")

check(abs((30000 * 16) / 12864 - 37.3134328358) < 1e-6 and "37.3" in tex,
      "30k updates at effective batch 16 equal 480k examples and 37.3 loader epochs")
check("reference half B" in tex and "reference half A" in tex and "source-motion-disjoint" in tex,
      "Table discloses the reference-half convention")

numeric_tokens = re.findall(r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)*(?:\\?%|[A-Za-z]+)?", tex)
evidence_files = sorted(RESULTS.glob("*.json"))
lines = [
    "# arXiv v6 numeric audit",
    "",
    f"- Manuscript: `{PAPER.relative_to(ROOT)}`",
    f"- Manuscript SHA-256: `{sha(PAPER)}`",
    f"- Numeric tokens extracted: **{len(numeric_tokens)}**",
    f"- Evidence assertions passed: **{len(checks)}**",
    "- Superseded 10k learned-model rows: **absent**",
    "- Reference convention: Table real row is half B; learned-row FVD uses half A; codec control is B reconstructed vs A.",
    "",
    "## Verified checks",
    "",
    *(f"- PASS — {item}" for item in checks),
    "",
    "## Evidence hashes",
    "",
    *(f"- `{path.relative_to(ROOT)}` — `{sha(path)}`" for path in evidence_files),
    "",
]
REPORT.write_text("\n".join(lines))
print(f"PASS: {len(checks)} assertions; {len(numeric_tokens)} numeric tokens; report={REPORT}")
