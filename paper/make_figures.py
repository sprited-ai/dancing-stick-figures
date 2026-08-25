"""Build the compact paper's quantitative figures from checked-in results."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "paper" / "results"
FIGS = ROOT / "paper" / "figs"
K1_RESULTS = ROOT / "pod_results" / "k1_final_eval_n64" / "correct_colored_cache"
PURPLE = "#6337C7"
TEAL = "#008C95"
ORANGE = "#E2762D"
GRAY = "#777777"


def load(name: str):
    with open(RESULTS / name) as f:
        return json.load(f)


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 180,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def image_benchmark(ax):
    entries = [
        ("UNet 64, 30k", "score_ia64.json"),
        ("DiT 64, 30k", "score_ib64.json"),
        ("UNet 64, 100k", "ia64L_scores.json"),
        ("DiT 64, 50k", "ib64L_scores.json"),
        ("UNet 128, 20k", "score_ia128.json"),
        ("DiT 128, 40k", "score_ib128.json"),
    ]
    labels, dtvr, dlie = [], [], []
    for label, filename in entries:
        d = load(filename)
        labels.append(label)
        dtvr.append(d["tvr"] - d["floor"]["tvr"])
        dlie.append(d["lie"] - d["floor"]["lie"])
    y = np.arange(len(labels))
    ax.axvline(0, color="#222222", lw=.7)
    ax.barh(y - .17, dtvr, height=.30, color=PURPLE, label=r"$\Delta$TVR")
    ax.barh(y + .17, dlie, height=.30, color=TEAL, label=r"$\Delta$LIE")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("excess error over matched real floor")
    ax.set_title("(a) Image approaches")
    ax.legend(ncol=2, loc="upper right")
    ax.grid(axis="x", color="#dddddd", lw=.5)


def video_benchmark(ax):
    scratch = load("eval_a64_final.json")
    warm = load("eval_a64i_final.json")
    ar = {
        "tvr": .149, "lie": .121, "head_jitter": .69, "angle_jerk": .19,
        "floor": {"tvr": .128, "lie": .103, "head_jitter": .57, "angle_jerk": .14},
    }
    models = [("8f scratch", scratch), ("8f image-init", warm), ("5s AR", ar)]
    metrics = [("TVR", "tvr"), ("LIE", "lie"), ("centroid", "head_jitter"), ("angle jerk", "angle_jerk")]
    x = np.arange(len(metrics))
    colors = [PURPLE, TEAL, ORANGE]
    for i, (name, d) in enumerate(models):
        ratios = [d[k] / d["floor"][k] for _, k in metrics]
        ax.bar(x + (i - 1) * .24, ratios, width=.22, label=name, color=colors[i])
    ax.axhline(1, color="#222222", lw=.8, ls="--")
    ax.set_xticks(x, [m[0] for m in metrics])
    ax.set_ylabel("model / real-floor error")
    ax.set_ylim(0, 1.9)
    ax.set_title("(b) Video approaches")
    ax.legend(ncol=1, loc="upper left")
    ax.grid(axis="y", color="#dddddd", lw=.5)


def corruption_heatmap(ax):
    d = load("corruption_500.json")["conditions"]
    keys = ["swap_LR_partial", "swap_LR_full", "stretch_bone", "delete_hand", "extra_arm"]
    labels = ["partial\nswap", "full\nswap", "stretch", "delete\nhand", "extra\narm"]
    metrics = [("TVR", "tvr"), ("LIE", "lie"), ("pixel MAE", "pixel_mae")]
    raw = np.array([[d[k][metric]["mean"] - d["real"][metric]["mean"] for k in keys]
                    for _, metric in metrics])
    scale = np.max(np.abs(raw), axis=1, keepdims=True)
    norm = np.divide(raw, scale, out=np.zeros_like(raw), where=scale > 0)
    ax.imshow(norm, cmap="PuBuGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(keys)), labels)
    ax.set_yticks(np.arange(len(metrics)), [m[0] for m in metrics])
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            text = f"{raw[i, j]:+.3f}"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if norm[i, j] > .58 else "#222222", fontsize=7)
    ax.set_title("(c) Controlled failures (delta from clean, n=500)")
    ax.tick_params(length=0)


def parse_val(path: Path):
    pattern = re.compile(r"step (\d+).*? val ([0-9.]+)$")
    points = []
    for line in path.read_text().splitlines():
        m = pattern.search(line)
        if m:
            points.append((int(m.group(1)), float(m.group(2))))
    return np.asarray(points)


def training_curve():
    fig, ax = plt.subplots(figsize=(3.35, 2.05))
    for name, path, color in [
        ("random init", ROOT / "out/video/a64/log.txt", PURPLE),
        ("image init", ROOT / "out/video/a64i/log.txt", TEAL),
    ]:
        p = parse_val(path)
        ax.plot(p[:, 0] / 1000, p[:, 1], marker="o", ms=2.2, lw=1.4, label=name, color=color)
    ax.scatter([10, 4], [.0112, .0100], color=[PURPLE, TEAL], zorder=5, s=18)
    ax.annotate("scratch @10k", (10, .0112), xytext=(11.5, .017),
                arrowprops={"arrowstyle": "-", "lw": .6}, fontsize=7)
    ax.annotate("image-init @4k", (4, .0100), xytext=(5.2, .026),
                arrowprops={"arrowstyle": "-", "lw": .6}, fontsize=7)
    ax.set_xlim(0, 32)
    ax.set_ylim(0, .065)
    ax.set_xlabel("video training steps (thousands)")
    ax.set_ylabel("validation v-loss")
    ax.set_title("Image initialization accelerates the early regime")
    ax.grid(color="#dddddd", lw=.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "warmstart_curve.pdf")
    fig.savefig(FIGS / "warmstart_curve.png")
    plt.close(fig)


def failure_modes():
    metrics = load("degenerate_120f_n128.json")["baselines"]
    distances = {name: row["fvd"] for name, row in metrics.items()}
    names = ["repeat_first", "shuffle_frames", "reverse_time", "loop_first_8"]
    labels = ["freeze", "shuffle", "reverse", "first 8\nlooped"]
    signals = [("centroid speed", "centroid_speed"), ("centroid accel.", "centroid_accel"),
               ("angle jerk", "angle_jerk")]
    colors = [PURPLE, TEAL, ORANGE]

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.15, 2.35), gridspec_kw={"wspace": .34})
    x = np.arange(len(names))
    for i, ((label, key), color) in enumerate(zip(signals, colors)):
        reference = metrics["real_reference_b"][key]["mean"]
        values = [metrics[name][key]["mean"] / reference for name in names]
        left.bar(x + (i - 1) * .24, values, width=.22, label=label, color=color)
    # A frozen video makes all three motion signals exactly zero.  Show the
    # zero values explicitly so the empty group is not mistaken for missing data.
    freeze_x = x[0] + np.array([-.24, 0, .24])
    left.scatter(freeze_x, np.zeros(3), marker="o", s=20, facecolors="white",
                 edgecolors=colors, linewidths=1.4, zorder=5, clip_on=False)
    left.text(x[0], .38, "all = 0", ha="center", va="bottom", fontsize=7.5, color="#333333")
    left.axhline(1, color="#222222", lw=.8, ls="--")
    left.set_xticks(x, labels)
    left.set_ylabel("signal / real reference")
    left.set_ylim(0, 11.2)
    left.set_title("(a) Controlled temporal failures")
    left.grid(axis="y", color="#dddddd", lw=.5)
    left.legend(ncol=1, loc="upper right")

    values = [distances[name] for name in names]
    bars = right.bar(x, values, color=[PURPLE, TEAL, ORANGE, "#B55AA5"])
    real_real = distances["real_reference_b"]
    right.axhline(real_real, color="#222222", lw=.9, ls="--", label=f"real--real: {real_real:.1f}")
    right.set_xticks(x, labels)
    right.set_ylabel("FVD to real reference A")
    right.set_ylim(0, max(values) * 1.18)
    right.set_title("(b) FVD sensitivity to temporal failures")
    right.grid(axis="y", color="#dddddd", lw=.5)
    right.legend(loc="upper left")
    for bar, value in zip(bars, values):
        right.text(bar.get_x() + bar.get_width() / 2, value + 8, f"{value:.0f}", ha="center", fontsize=7)

    fig.savefig(FIGS / "failure_modes.pdf")
    fig.savefig(FIGS / "failure_modes.png")
    plt.close(fig)


def k1_warmstart_tradeoff():
    paired = json.loads((K1_RESULTS / "correct_paired_comparison_n64.json").read_text())["metrics"]
    real = json.loads((K1_RESULTS / "correct_real_floor_n64.json").read_text())["oracle"]
    panels = [
        ("a  Frame structure and occupancy", [("TVR", "tvr"), ("LIE", "lie"),
                                               ("colour purity", "cpe"), ("foreground", "fg")]),
        ("b  Temporal behaviour", [("centroid speed", "centroid_speed"),
                                    ("centroid accel.", "centroid_accel"),
                                    ("motion fraction", "motion_fraction"),
                                    ("angle jerk", "angle_jerk")]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.35), gridspec_kw={"wspace": .30})
    for ax, (title, metrics) in zip(axes, panels):
        x = np.arange(len(metrics))
        scratch = [paired[key]["scratch"] / real[key] for _, key in metrics]
        warm = [paired[key]["warm"] / real[key] for _, key in metrics]
        ax.bar(x - .18, scratch, width=.36, color=PURPLE, label="scratch")
        ax.bar(x + .18, warm, width=.36, color=TEAL, label="T2I warm-start")
        ax.axhline(1, color="#222222", lw=.8, ls="--", label="real reference")
        ax.set_xticks(x, [label for label, _ in metrics])
        ax.tick_params(axis="x", labelrotation=12)
        ax.set_ylabel("generated / real-reference value")
        ax.set_title(title)
        ax.grid(axis="y", color="#dddddd", lw=.5)
    axes[0].set_ylim(0, 5.6)
    axes[1].set_ylim(0, 4.1)
    axes[0].legend(ncol=1, loc="upper right")
    fig.savefig(FIGS / "k1_warmstart_tradeoff.pdf")
    fig.savefig(FIGS / "k1_warmstart_tradeoff.png")
    plt.close(fig)


def m6_milestone_tradeoff():
    data = load("m6_h8_milestones_n64.json")
    real = data["real_reference"]
    steps = sorted(int(s) for s in data["milestones"])
    series = {step: data["milestones"][str(step)] for step in steps}
    r0 = data.get("r0_reference")
    panels = [
        ("a  Frame structure", [("TVR", "tvr"), ("LIE", "lie"), ("colour purity", "cpe")]),
        ("b  Motion calibration", [("centroid speed", "centroid_speed"),
                                   ("motion fraction", "motion_fraction"),
                                   ("height variance", "height_var")]),
    ]
    long_run = data.get("h8_100k_run")
    long_steps = sorted(int(s) for s in long_run["milestones"]) if long_run else []
    long_series = {step: long_run["milestones"][str(step)] for step in long_steps} if long_run else {}
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.35), gridspec_kw={"wspace": .30})
    colors = [PURPLE, TEAL, ORANGE]
    for ax, (title, metrics) in zip(axes, panels):
        for color, (label, key) in zip(colors, metrics):
            ax.plot(steps, [series[s][key] / real[key] for s in steps],
                    marker="o", ms=3, lw=1.2, color=color, label=label)
            if long_series:
                ax.plot(long_steps, [long_series[s][key] / real[key] for s in long_steps],
                        marker="s", ms=3, lw=1.2, ls=":", color=color, mfc="white")
            if r0 is not None:
                ax.plot([steps[-1]], [r0[key] / real[key]], marker="D", ms=4,
                        color=color, mec="#222222", mew=.6, ls="none")
        if long_series:
            ax.set_xscale("log")
        ax.axhline(1, color="#222222", lw=.8, ls="--")
        ax.set_xlabel("H8 block-AR training step")
        ax.set_ylabel("generated / real value")
        ax.set_title(title)
        ax.grid(axis="y", color="#dddddd", lw=.5)
        ax.legend(loc="upper right")
    if r0 is not None:
        axes[1].plot([], [], marker="D", ms=4, color="#999999", mec="#222222",
                     mew=.6, ls="none", label="R0 full-clip 10k")
    if long_series:
        axes[1].plot([], [], marker="s", ms=3, lw=1.2, ls=":", color="#999999",
                     mfc="white", label="independent 100k run")
    if r0 is not None or long_series:
        axes[1].legend(loc="upper right")
    fig.savefig(FIGS / "m6_milestone_tradeoff.pdf")
    fig.savefig(FIGS / "m6_milestone_tradeoff.png")
    plt.close(fig)


def m6_fix_comparison():
    data = load("m6_h8_milestones_n64.json")
    real = data["real_reference"]
    h8 = data["h8_100k_run"]["milestones"]
    v8 = data["v8_100k_run"]["milestones"]
    h8_steps = sorted(int(s) for s in h8)
    v8_steps = sorted(int(s) for s in v8)
    long_run = data.get("v8_300k_run")
    long_steps = sorted(int(s) for s in long_run["milestones"]) if long_run else []
    panels = [
        ("a  Frame structure (TVR)", "tvr"),
        ("b  Motion (centroid speed)", "centroid_speed"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.2), gridspec_kw={"wspace": .28})
    for ax, (title, key) in zip(axes, panels):
        ax.plot(h8_steps, [h8[str(s)][key] / real[key] for s in h8_steps],
                marker="o", ms=3, lw=1.2, color=PURPLE, label="h8 baseline")
        ax.plot(v8_steps, [v8[str(s)][key] / real[key] for s in v8_steps],
                marker="s", ms=3, lw=1.4, color=TEAL, label="v8 combined fix")
        if long_run:
            ax.plot(long_steps, [long_run["milestones"][str(s)][key] / real[key] for s in long_steps],
                    marker="^", ms=3, lw=1.2, ls=":", color=TEAL, mfc="white",
                    label="v8 at 3x budget" if key == "tvr" else None)
        ax.axhline(1, color="#222222", lw=.8, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("training step")
        ax.set_ylabel("generated / real value")
        ax.set_title(title)
        ax.grid(axis="y", color="#dddddd", lw=.5)
        ax.legend(loc="upper right")
    fig.savefig(FIGS / "m6_fix_comparison.pdf")
    fig.savefig(FIGS / "m6_fix_comparison.png")
    plt.close(fig)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    style()
    fig = plt.figure(figsize=(7.15, 4.75))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=.38, wspace=.34)
    image_benchmark(fig.add_subplot(gs[0, 0]))
    video_benchmark(fig.add_subplot(gs[0, 1]))
    corruption_heatmap(fig.add_subplot(gs[1, :]))
    fig.savefig(FIGS / "benchmark_summary.pdf")
    fig.savefig(FIGS / "benchmark_summary.png")
    plt.close(fig)
    training_curve()
    failure_modes()
    k1_warmstart_tradeoff()
    m6_milestone_tradeoff()
    m6_fix_comparison()


if __name__ == "__main__":
    main()
