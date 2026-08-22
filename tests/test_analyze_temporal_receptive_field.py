from scripts.analyze_temporal_receptive_field import report


def test_temporal_receptive_field_audit_has_stable_expected_spans():
    result = report()
    dsf = result["dsf_f8t4d16"]
    assert dsf["encoder_raw_frame_span"]["maximum_tail_span"] == 91
    assert dsf["decoder_latent_step_span"]["minimum_tail_span"] == 23
    assert dsf["decoder_latent_step_span"]["maximum_tail_span"] == 24
    assert dsf["decoder_video_frame_equivalent_span"] == {"minimum": 92, "maximum": 96}
    spatial = dsf["decoder_spatiotemporal_probe"]
    assert spatial["latent_temporal_steps"] == 24
    assert spatial["total_latent_cells"] == 24 * 8 * 8
    assert spatial["all_temporal_lags_cover_full_8x8"]


def test_reference_horizons_exceed_per_layer_cache_depth():
    result = report()
    for key in ("wan2.1_public_code", "cogvideox_diffusers_public_code"):
        codec = result[key]
        assert codec["per_causal_conv_feature_cache"] == 2
        assert codec["decoder_latent_step_span_inferred"]["minimum_tail_span"] > 2
