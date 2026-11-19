"""D4 current-frame control with the original GRU layers and no temporal state."""

from .temporal_model import CausalJointRefiner


MODEL_ID = "current_frame"
CONTROL_VERSION = "r4_current_frame_v1"


class CurrentFrameRefiner(CausalJointRefiner):
    """Use the ordinary refiner's current observations independently per frame.

    Layer names, shapes and initialization are inherited unchanged. Supplied
    state/reset arguments never reach the parent: every call starts with zero
    hidden, heatmap and in/out history. Returning None also prevents callers
    from retaining a prediction state across steps or chunk boundaries.
    """

    def step(self, features, raw_heatmap, raw_inout, bboxes, state=None, reset=None):
        heatmap, score, _ = super().step(
            features, raw_heatmap, raw_inout, bboxes, state=None, reset=None)
        return heatmap, score, None


def parameter_usage(model):
    """Describe retained parameter shapes and structurally unused weight paths.

    These counts are neither independent effective degrees of freedom nor
    measured FLOPs. The dense GRU calculation is retained. With zero previous
    hidden, recurrent weights have zero task gradient, but recurrent biases
    can still affect a single step. AdamW decay can move zero-gradient weights.
    """
    hidden = model.hidden_dim
    rows = 3 * hidden
    previous_map_first = 3 * hidden + 8 * 8
    previous_map_stop = previous_map_first + 8 * 8
    previous_score_column = 3 * hidden + 2 * 8 * 8 + 4 + 1
    groups = {
        "gru.weight_hh": {
            "elements": model.gru.weight_hh.numel(),
            "reason": "previous_hidden_is_zero_for_every_frame",
        },
        "gru.weight_ih.previous_heatmap_columns": {
            "elements": rows * 8 * 8,
            "input_columns_start": previous_map_first,
            "input_columns_stop_exclusive": previous_map_stop,
            "reason": "previous_heatmap_input_is_always_zero",
        },
        "gru.weight_ih.previous_inout_column": {
            "elements": rows,
            "input_column": previous_score_column,
            "reason": "previous_inout_input_is_always_zero",
        },
    }
    return {
        "full_parameter_count": sum(value.numel() for value in model.parameters()),
        "structural_zero_weight_elements": sum(item["elements"] for item in groups.values()),
        "structural_zero_weight_groups": groups,
        "same_layer_names_and_shapes_as_ordinary_gru": True,
        "effective_parameter_use_matches_recurrent_model": False,
        "interpretation": "retained_parameter_shapes; not_effective_degrees_of_freedom_or_FLOPs",
        "bias_note": "GRU biases are retained; candidate recurrent bias can affect output even with zero hidden",
        "optimizer_note": "Zero task-gradient weights can still change under AdamW weight decay",
    }
