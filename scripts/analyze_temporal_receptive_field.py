"""Audit steady-state temporal dependency spans of DSF and reference codecs.

The calculation is structural: every sequence element carries the set of
original time indices that can influence it.  Causal convolutions, residual
blocks, block-end subsampling, and temporal replication propagate those sets.
It deliberately reports *theoretical dependency span*, which is distinct from
temporal compression, per-layer cache depth, and the operational window used by
a tiled/streaming implementation.

Reference architecture descriptors are transcribed from the public code pinned
in ``REFERENCE_REVISIONS``.  Their values are labelled inferred rather than
published metrics because the papers do not report receptive-field horizons.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


REFERENCE_REVISIONS = {
    "wan2.1": {
        "revision": "9737cba9c1c3c4d04b33fcad41c111989865d315",
        "source": "https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/vae.py",
        "note": "dim_mult=[1,2,4,4], two residual blocks/stage, two temporal 2x stages",
    },
    "cogvideox_diffusers": {
        "revision": "0a0c57d3889e0184784ff507e1d795e2a3719be6",
        "source": "https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl_cogvideox.py",
        "note": "four stages, three encoder/four decoder residual blocks per stage, temporal ratio 4",
    },
}


Dependency = frozenset[int]
Coordinate = tuple[int, int, int]


def source(length: int) -> list[Dependency]:
    return [frozenset((index,)) for index in range(length)]


def causal_conv(sequence: list[Dependency], kernel: int = 3) -> list[Dependency]:
    """Stride-one left-padded causal convolution."""
    result = []
    for index in range(len(sequence)):
        deps: set[int] = set()
        for previous in range(index - kernel + 1, index + 1):
            if previous >= 0:
                deps.update(sequence[previous])
        result.append(frozenset(deps))
    return result


def residual_blocks(sequence: list[Dependency], count: int) -> list[Dependency]:
    for _ in range(count):
        residual = sequence
        hidden = causal_conv(causal_conv(sequence))
        sequence = [left | right for left, right in zip(residual, hidden)]
    return sequence


def block_end_downsample(sequence: list[Dependency], factor: int) -> list[Dependency]:
    return sequence if factor == 1 else sequence[factor - 1 :: factor]


def causal_pair_downsample(sequence: list[Dependency]) -> list[Dependency]:
    """Steady causal 2x stage: each emitted step sees the previous/current pair."""
    result = []
    for end in range(1, len(sequence), 2):
        deps: set[int] = set()
        for index in range(max(0, end - 2), end + 1):
            deps.update(sequence[index])
        result.append(frozenset(deps))
    return result


def first_then_pair_average(sequence: list[Dependency]) -> list[Dependency]:
    """CogVideoX's first-frame-preserving temporal average pooling."""
    result = [sequence[0]]
    for start in range(1, len(sequence) - 1, 2):
        result.append(sequence[start] | sequence[start + 1])
    return result


def repeat(sequence: list[Dependency], factor: int) -> list[Dependency]:
    return [deps for item in sequence for deps in (item,) * factor]


def span(sequence: list[Dependency], tail: int = 8) -> dict[str, int | list[int]]:
    widths = []
    counts = []
    for deps in sequence[-tail:]:
        widths.append(max(deps) - min(deps) + 1)
        counts.append(len(deps))
    return {
        "minimum_tail_span": min(widths),
        "maximum_tail_span": max(widths),
        "minimum_tail_dependency_count": min(counts),
        "maximum_tail_dependency_count": max(counts),
        "tail_spans": widths,
    }


def reverse_causal_conv3d(
    points: set[Coordinate], *, input_shape: tuple[int, int, int],
    kernel: tuple[int, int, int] = (3, 3, 3),
) -> set[Coordinate]:
    """Map selected outputs back through a causal 3D convolution."""
    time, height, width = input_shape
    kt, kh, kw = kernel
    result = set()
    for out_t, out_y, out_x in points:
        for source_t in range(out_t - kt + 1, out_t + 1):
            for source_y in range(out_y - kh // 2, out_y + kh // 2 + 1):
                for source_x in range(out_x - kw // 2, out_x + kw // 2 + 1):
                    if 0 <= source_t < time and 0 <= source_y < height and 0 <= source_x < width:
                        result.add((source_t, source_y, source_x))
    return result


def reverse_residual_blocks(
    points: set[Coordinate], *, shape: tuple[int, int, int], count: int,
) -> set[Coordinate]:
    # Two kernel-3 convolutions include the identity coordinate, so their
    # dependency set already contains the residual shortcut's dependency.
    for _ in range(count):
        points = reverse_causal_conv3d(
            reverse_causal_conv3d(points, input_shape=shape), input_shape=shape,
        )
    return points


def reverse_causal_upsample(
    points: set[Coordinate], *, input_shape: tuple[int, int, int], scale: tuple[int, int, int],
) -> set[Coordinate]:
    """Map through causal smoothing convolution, then nearest-neighbour repeat."""
    time, height, width = input_shape
    st, sh, sw = scale
    output_shape = (time * st, height * sh, width * sw)
    points = reverse_causal_conv3d(points, input_shape=output_shape)
    return {(t // st, y // sh, x // sw) for t, y, x in points}


def dsf_decoder_spatiotemporal_map(
    *, latent_frames: int = 30, latent_size: int = 8, blocks_per_stage: int = 2,
) -> dict:
    """Trace a central late output pixel back to f8t4 latent coordinates."""
    output_frames = latent_frames * 4
    points: set[Coordinate] = {(output_frames - 1, 32, 32)}
    points = reverse_causal_conv3d(points, input_shape=(output_frames, 64, 64))
    points = reverse_residual_blocks(
        points, shape=(output_frames, 64, 64), count=blocks_per_stage,
    )
    points = reverse_causal_upsample(
        points, input_shape=(output_frames, 32, 32), scale=(1, 2, 2),
    )
    points = reverse_residual_blocks(
        points, shape=(output_frames, 32, 32), count=blocks_per_stage,
    )
    points = reverse_causal_upsample(
        points, input_shape=(latent_frames * 2, 16, 16), scale=(2, 2, 2),
    )
    points = reverse_residual_blocks(
        points, shape=(latent_frames * 2, 16, 16), count=blocks_per_stage,
    )
    points = reverse_causal_upsample(
        points, input_shape=(latent_frames * 2, latent_size, latent_size), scale=(1, 2, 2),
    )
    points = reverse_residual_blocks(
        points, shape=(latent_frames * 2, latent_size, latent_size), count=blocks_per_stage,
    )
    points = reverse_causal_upsample(
        points, input_shape=(latent_frames, latent_size, latent_size), scale=(2, 1, 1),
    )
    points = reverse_residual_blocks(
        points, shape=(latent_frames, latent_size, latent_size), count=blocks_per_stage,
    )
    newest = max(t for t, _, _ in points)
    by_lag: dict[str, dict] = {}
    for latent_t in sorted({t for t, _, _ in points}, reverse=True):
        cells = {(y, x) for t, y, x in points if t == latent_t}
        ys, xs = zip(*cells)
        by_lag[str(newest - latent_t)] = {
            "spatial_cells": len(cells),
            "fraction_of_8x8_grid": len(cells) / (latent_size * latent_size),
            "bounding_box_yx": [min(ys), min(xs), max(ys), max(xs)],
        }
    return {
        "probe_output_tyx": [output_frames - 1, 32, 32],
        "latent_temporal_steps": len(by_lag),
        "total_latent_cells": len(points),
        "all_temporal_lags_cover_full_8x8": all(
            item["spatial_cells"] == latent_size * latent_size for item in by_lag.values()
        ),
        "by_latent_lag": by_lag,
    }


def dsf_f8t4(blocks_per_stage: int = 2, length: int = 2048) -> dict:
    enc = causal_conv(source(length))
    enc = residual_blocks(enc, blocks_per_stage)
    enc = causal_conv(enc)  # f2 spatial downsample, temporal kernel 3
    enc = residual_blocks(enc, blocks_per_stage)
    enc = causal_conv(enc)  # f4 spatial downsample, temporal kernel 3
    enc = block_end_downsample(enc, 2)
    enc = residual_blocks(enc, blocks_per_stage)
    # f8 spatial downsample has temporal kernel 1.
    enc = residual_blocks(enc, blocks_per_stage)
    enc = causal_conv(enc)  # explicit temporal refinement
    enc = block_end_downsample(enc, 2)
    enc = residual_blocks(enc, blocks_per_stage)

    dec = source(length)
    dec = residual_blocks(dec, blocks_per_stage)
    dec = causal_conv(repeat(dec, 2))
    dec = residual_blocks(dec, blocks_per_stage)
    dec = residual_blocks(dec, blocks_per_stage)
    dec = causal_conv(repeat(dec, 2))
    dec = residual_blocks(dec, blocks_per_stage)
    dec = causal_conv(dec)  # spatial upsample still applies a temporal kernel
    dec = residual_blocks(dec, blocks_per_stage)
    dec = causal_conv(dec)
    return {
        "encoder_raw_frame_span": span(enc),
        "decoder_latent_step_span": span(dec),
        "decoder_video_frame_equivalent_span": {
            "minimum": span(dec)["minimum_tail_span"] * 4,
            "maximum": span(dec)["maximum_tail_span"] * 4,
        },
        "per_causal_conv_feature_cache": 2,
        "training_window_video_frames": 20,
        "m6_ground_truth_encode_window_video_frames": 24,
        "current_sliding_decode_context_latents": 5,
        "current_sliding_decode_context_video_frames": 20,
        "decoder_spatiotemporal_probe": dsf_decoder_spatiotemporal_map(
            blocks_per_stage=blocks_per_stage,
        ),
        "m6_denoiser_connectivity": {
            "block_0": "global spatial attention independently inside every latent frame",
            "block_1": "temporal attention at matched spatial indices; target queries read all clean-history and target times",
            "after_blocks_0_and_1": "every target token has a dependency path to every spatial cell in every visible history/target latent frame",
            "qualification": "same graph coverage as full spatiotemporal attention after two blocks, but not the same one-layer interaction or inductive bias",
        },
    }


def wan21(length: int = 2048) -> dict:
    enc = causal_conv(source(length))
    for stage in range(4):
        enc = residual_blocks(enc, 2)
        if stage in (1, 2):
            enc = causal_pair_downsample(enc)
    enc = residual_blocks(enc, 2)
    enc = causal_conv(enc)

    dec = causal_conv(source(length))
    dec = residual_blocks(dec, 2)
    for stage in range(4):
        dec = residual_blocks(dec, 3)
        if stage in (0, 1):
            # Wan applies a causal temporal convolution, channel-to-time
            # rearrangement, then emits two frames per latent step.
            dec = repeat(causal_conv(dec), 2)
    dec = causal_conv(dec)
    return {
        "encoder_raw_frame_span_inferred": span(enc),
        "decoder_latent_step_span_inferred": span(dec),
        "decoder_video_frame_equivalent_span_inferred": {
            "minimum": span(dec)["minimum_tail_span"] * 4,
            "maximum": span(dec)["maximum_tail_span"] * 4,
        },
        "per_causal_conv_feature_cache": 2,
        "streaming_strategy": "persistent feature cache; encode 1 then 4-frame chunks; decode one latent at a time",
    }


def cogvideox(length: int = 2048) -> dict:
    enc = causal_conv(source(length))
    for stage in range(4):
        enc = residual_blocks(enc, 3)
        if stage in (0, 1):
            enc = first_then_pair_average(enc)
    enc = residual_blocks(enc, 2)
    enc = causal_conv(enc)

    dec = causal_conv(source(length))
    dec = residual_blocks(dec, 2)
    for stage in range(4):
        dec = residual_blocks(dec, 4)
        if stage in (0, 1):
            dec = repeat(dec, 2)
    dec = causal_conv(dec)
    return {
        "encoder_raw_frame_span_inferred": span(enc),
        "decoder_latent_step_span_inferred": span(dec),
        "decoder_video_frame_equivalent_span_inferred": {
            "minimum": span(dec)["minimum_tail_span"] * 4,
            "maximum": span(dec)["maximum_tail_span"] * 4,
        },
        "per_causal_conv_feature_cache": 2,
        "streaming_strategy": "rolling convolution caches in the public implementation",
    }


def report() -> dict:
    return {
        "definitions": {
            "dependency_span": "oldest-to-newest source indices that can affect a steady-state output",
            "not_compression": "a t4 codec can have a receptive span far larger than four frames",
            "not_runtime_memory": "persistent per-layer caches can preserve a long composed span while retaining only two feature frames per kernel-3 layer",
        },
        "dsf_f8t4d16": dsf_f8t4(),
        "wan2.1_public_code": wan21(),
        "cogvideox_diffusers_public_code": cogvideox(),
        "reference_revisions": REFERENCE_REVISIONS,
        "limitations": [
            "Wan and CogVideoX spans are static dependency-graph inferences, not values reported by their authors.",
            "First-frame boundary behavior differs from steady state by design.",
            "LTX-Video is omitted from exact span arithmetic because checkpoint block metadata changes the configurable public autoencoder graph.",
            "A theoretical dependency does not measure how strongly trained weights use the oldest frame.",
            "The spatial probe is for a central late output pixel; finite-grid boundary probes can have smaller sets.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="output/codec_temporal_receptive_field.json")
    args = parser.parse_args()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = report()
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
