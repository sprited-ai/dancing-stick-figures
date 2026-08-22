from scripts.compare_vae_codecs import compare


def _short(rgba=.001, edge=.01, iou=.98):
    return {"metrics": {"rgba_l1": rgba, "rgb_support_l1": .01,
                        "rgb_edge_l1": edge, "alpha_iou_0.5": iou,
                        "white_composite_psnr": 40.0}}


def _long(rgba=.001, seam=.01, within=.01, seconds=.1):
    return {"metrics": {"sliding_rgba_l1": rgba, "sliding_white_psnr": 40.0,
                        "sliding_boundary_transition_l1": seam,
                        "sliding_within_transition_l1": within,
                        "causal_prefix_max_abs": 0.0},
            "timing": {"sliding_seconds": seconds}}


def test_all_declared_gates_must_pass():
    protocol = {"version": 1, "baseline": "corrected f8t2d32 step 40000",
                "candidate_hard_gates": {
        "short_rgba_l1_max": .0025, "short_rgb_edge_l1_max": .025,
        "short_alpha_iou_min": .95, "long_sliding_rgba_l1_max": .0025,
        "long_sliding_seam_to_within_ratio_max": 1.1,
        "causal_prefix_max_abs_max": 1e-5,
        "sliding_runtime_ratio_vs_baseline_max": 2.0}}
    passed = compare(protocol, _short(), _long(), _short(), _long())
    assert passed["quantitative_gate_pass"]
    assert passed["selection"] == "pending_visual_audit"
    failed = compare(protocol, _short(), _long(), _short(edge=.03), _long())
    assert not failed["quantitative_gate_pass"]
    assert failed["selection"] == protocol["baseline"]
