from copy import deepcopy


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

# Training only. Select any of: hflip, brightness_contrast, scale.
# Keep the factory defaults separate so legacy resumes do not enable augmentation.
DEFAULT_TRAIN_AUGMENTATION = {
    "enabled": [],
    "hflip_p": 0.5,
    "brightness_contrast_p": 0.5,
    "brightness_range": [0.85, 1.15],
    "contrast_range": [0.85, 1.15],
    "zoom_out_p": 0.4,
    "zoom_in_p": 0.1,
    "zoom_out_range": [0.6, 1.0],
    "zoom_in_range": [1.0, 1.2],
}

# Edit this dictionary (including numeric parameters) for new runs.
train_augmentation = deepcopy(DEFAULT_TRAIN_AUGMENTATION)


def validate_seg_input(value):
    if value not in SEG_INPUT_MODES:
        raise ValueError(f"seg_input must be one of {SEG_INPUT_MODES}, got {value!r}.")
    return value
