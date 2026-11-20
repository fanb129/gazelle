"""Predeclared development budget and selection rules; not experiment results."""


def development_selection_plan():
    """Freeze a small, shared search budget before training temporal heads."""
    return {
        "version": "r1_selection_v1",
        "status": "planned_not_executed",
        "data": {
            "fit": "development_train_only",
            "select": "grouped_validation_only",
            "test": "no_structure_threshold_or_checkpoint_selection",
            "base_checkpoint": "already_trained_on_official_VAT_train; shared_by_all_methods",
        },
        "training": {
            "screening_seed": 0, "confirmation_seeds": [0, 1, 2],
            "learning_rates": [0.0001, 0.0003], "lambda_state": [0.1, 1.0],
            "max_screening_configs_per_trainable_variant": 4,
            "hidden_dimension": 128, "clip_frames": 16, "batch_clips": 8,
            "maximum_epochs": 20, "early_stopping_patience": 4,
            "loss": "GT_inframe_heatmap_MSE + lambda_state * valid_frame_state_BCE",
            "trainable": "new_projection_temporal_module_and_refinement_heads_only",
            "history": "predicted_only; empty_at_contiguous_track_start; no_GT_teacher_forcing",
            "confirmation": "same_selected_hyperparameters_for_each_seed; report_all_seeds",
        },
        "simple_controls": {
            "fixed_heatmap_ema_alpha": [0.11920292202211755, 0.25, 0.5, 0.75],
            "rule_weights_reference": [-2, 20, 2, 10],
            "calibrated_rule_max_candidates": 16,
            "one_euro_max_candidates": 4,
            "speed_adaptive_heatmap_ema_max_candidates": 4,
            "candidate_values": "declare_in_R4_config_before_evaluating_that_family",
            "coordinate_filter": "raw_heatmap_AUC_and_raw_inout_AP_retained_and_labelled",
        },
        "selection": {
            "reference": "single_frame_on_identical_validation_records",
            "accuracy_tolerances_absolute": {"auc_drop": 0.002, "l2_increase": 0.002,
                                              "inout_ap_drop": 0.002},
            "response_floor": "gaze_shift_joint_and_all_state_response_rates_not_below_reference",
            "undefined_response": "not_used_as_a_selection_gate_when_no_assessable_reference_events",
            "rank_eligible": ["accurate_stable_ratio_desc", "joint_gaze_shift_response_rate_desc",
                              "state_response_rate_desc", "inout_ap_desc", "l2_asc", "auc_desc",
                              "config_id_lexicographic"],
            "no_eligible_candidate": "retain_single_frame_reference; report_all_candidates_and_Pareto_front",
            "conditional_latency": "paired_common_events_only; not_ranked_by_unpaired_responder_means",
            "matched_jitter_absolute_tolerance": 0.001,
            "matched_accurate_stability_absolute_tolerance": 0.01,
            "full_frontier": "report_accuracy_stability_response_tradeoffs_for_all_evaluated_candidates",
        },
        "statistics": {
            "unit": "source_video_group; never_independent_person_frames",
            "paired_bootstrap_replicates": 2000, "bootstrap_seed": 20261117,
            "resampling": "same_sampled_groups_for_all_methods; recompute_pooled_metrics_from_predictions",
            "latency": "resample_groups_then_pair_same_events; report_response_and_censoring",
        },
        "manual_diagnosis": {
            "partition": "development_train_only",
            "status": "unreviewed_until_RGB_and_prediction_inspection",
            "proxy_events": "not_target_identity_or_history_validity_ground_truth",
            "completion": "all_selected_cases_reviewed; preserve_unresolvable_labels_and_notes",
        },
        "changes": "version_and_record_any_amendment_before_training; never_amend_from_test_results",
    }
