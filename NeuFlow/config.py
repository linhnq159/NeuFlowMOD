feature_dim_s16 = 128
context_dim_s16 = 64
iter_context_dim_s16 = 64
feature_dim_s8 = 128
context_dim_s8 = 64
iter_context_dim_s8 = 64
feature_dim_s1 = 128

# Segmentation head input: "flow" (u, v) or "flow_rgb" (u, v, current R, G, B).
seg_input = "flow"
SEG_INPUT_MODES = ("flow", "flow_rgb")


def validate_seg_input(value):
    if value not in SEG_INPUT_MODES:
        raise ValueError(f"seg_input must be one of {SEG_INPUT_MODES}, got {value!r}.")
    return value
