# Training from image and mask folders

## Quick start: MOD_ONESHOT

Run the following commands from the repository root. Activate a Python
environment with PyTorch and OpenCV; this machine has the `SEA-RAFT` environment:

```bash
cd /media/linhnq/Data5/Project/NeuFlow_MOD_1
conda activate SEA-RAFT
```

Training and PyTorch inference support **PyTorch 2.3+** with the matching
torchvision build (for example, PyTorch 2.3 / torchvision 0.18). Check the active
environment before running:

```bash
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Checkpoint loading uses `weights_only=True` on both old and new PyTorch.
PyTorch 2.4+ additionally supports older checkpoint metadata stored as
`argparse.Namespace`; PyTorch 2.3 supports the tensor/dictionary checkpoints
written by this trainer and the bundled NeuFlow pretrained weights.

TensorBoard is optional. If neither `tensorboard` nor `tensorboardX` is installed,
the trainer prints a message, disables TensorBoard and continues with CSV/JSONL,
`train.log` and checkpoints. Use `--no-tensorboard` to disable it explicitly, or
`python -m pip install tensorboard` to enable it in the active environment.

Train with both frames and the current-frame mask resized to **960×544**:

```bash
python tool/train.py \
  --data_path datasets/MOD_ONESHOT \
  --resize 960 544 \
  --batch_size 8 \
  --num_workers 8 \
  --frame_skip 1 \
  --val_split 0.2 \
  --device cuda:0 \
  --no-amp
```

To keep frames at their original **1920×1080**, omit `--resize`. Only masks
resize to 1920×1080. Choose a batch size that fits GPU memory; training requires
at least two samples per batch:

```bash
python tool/train.py \
  --data_path datasets/MOD_ONESHOT \
  --batch_size 2 \
  --num_workers 8 \
  --frame_skip 1 \
  --val_split 0.2 \
  --device cuda:0
```

`--img_size 960 544` is equivalent to `--resize 960 544`. Images belong under
`datasets/MOD_ONESHOT/images/` and masks under `datasets/MOD_ONESHOT/masks/`,
with matching sequence folders and filename stems. Only the current frame
needs a mask; unmasked frames can still be used as past frames. `frame_skip=1`
pairs adjacent images, and `val_split=0.2` reserves the last 20% of each sequence
for validation. Use `--val_split 0` to disable validation.

The trainer prints the checkpoint directory at startup and saves
`checkpoint/last.pth` after every epoch under
`work_seg/checkpoint/<run_name>/`. Resume a resized run with its actual checkpoint:

```bash
python tool/train.py \
  --data_path datasets/MOD_ONESHOT \
  --resize 960 544 \
  --resume /path/to/run/checkpoint/last.pth \
  --device cuda:0
```

Repeat the original resize, batch, frame-skip and other training settings when
resuming. For a run using original-size frames, omit `--resize` on resume too.
To inspect all available options, run `python tool/train.py --help`.

## Optional training augmentation

Augmentation is **off by default**. Enable all three transforms for a new run:

```bash
python tool/train.py \
  --data_path datasets/MOD_ONESHOT \
  --resize 480 272 \
  --batch_size 4 \
  --num_workers 6 \
  --frame_skip 3 \
  --val_split 0.2 \
  --augment hflip brightness_contrast scale \
  --device cuda:0 \
  --no-amp
```

Select any subset, such as `--augment hflip brightness_contrast`, or use
`--augment none` to disable everything explicitly. Numeric settings are editable
in `train_augmentation` in `NeuFlow/config.py`; for example:

```python
train_augmentation = deepcopy(DEFAULT_TRAIN_AUGMENTATION)
train_augmentation.update(
    enabled=["hflip", "brightness_contrast", "scale"],
    hflip_p=0.5,
    brightness_contrast_p=0.5,
    brightness_range=[0.85, 1.15],
    contrast_range=[0.85, 1.15],
    zoom_out_p=0.4,
    zoom_in_p=0.1,
    zoom_out_range=[0.6, 1.0],
    zoom_in_range=[1.0, 1.2],
)
```

Edit `train_augmentation`, leaving `DEFAULT_TRAIN_AUGMENTATION` as the factory
defaults. New runs (including `--transfer`) use this config when `--augment` is
omitted. Resume restores all numeric settings and enabled modes from the
checkpoint; explicit `--augment` replaces only its enabled modes. Legacy
checkpoints without augmentation metadata default to disabled on resume.
The effective dictionary is printed at startup and saved in `train.log`, run
and session config JSON, and checkpoint `args.train_augmentation`.

Transforms run **after the existing resize and stabilization**, in the order
**scale/crop → horizontal flip → brightness/contrast**. Both RGB inputs always
use the same geometric parameters; the current-frame mask follows the same
geometry. Output size remains the chosen `--resize`, or native frame size if
resize is omitted. Samples within a batch must still have equal sizes.

| Enabled transform | Default probability | Effect on the two frames | Effect on current mask |
| --- | --- | --- | --- |
| `scale`: zoom out | 40% | Shrink by one shared factor in `[0.6, 1.0]`, paste at the same random position on the original-size canvas | Same shrink/position, background padding is 0 |
| `scale`: zoom in | 10% | Enlarge by one shared factor in `[1.0, 1.2]`, crop the same random window back to the original size | Same enlargement and crop |
| `scale`: unchanged | 50% | Keep original scale and field of view | Unchanged |
| `hflip` | 50% | Flip both horizontally | Flip horizontally |
| `brightness_contrast` | 50% | Shared brightness and contrast factors, each sampled independently from `[0.85, 1.15]` | Unchanged |

The three scale branches are mutually exclusive; flip and photometric decisions
are independent. Zoom-out padding uses one shared RGB color: the average of
the two frames' per-channel means before augmentation. It does not reflect or
repeat objects. Contrast is adjusted around 127.5, followed by multiplication
by the brightness factor and clipping to `[0, 255]`. RGB resizing is bilinear;
mask resizing uses nearest-neighbor and preserves binary labels.

For example, scale `0.6` makes a 20-pixel-wide object approximately 12 pixels
wide and its area approximately 36% of the original. Zoom-in makes objects
larger but **cannot restore details already lost during input resizing**.
Zoom-in probability must be lower than zoom-out probability whenever zoom-in
is enabled, and their sum must be at most 1. Set `zoom_in_p=0` for zoom-out only;
both probabilities may be zero. Invalid probabilities/ranges are rejected.

**Small-object protection:** before scale/crop, label every foreground region
with 8-connected components. Transform these IDs with the same nearest-neighbor
operation as the target. If **any** region disappears completely, discard only
scale/crop for both frames and the mask, without retrying. Flip and
brightness/contrast may still run. A partially cropped region is accepted if
at least one pixel remains. Empty masks can be augmented normally.

This guard protects regions present **after the ordinary input resize**; it
cannot recover regions that resize already removed or guarantee that a surviving
one-pixel object is easy to recognize. Rejected transforms mean the actual
accepted zoom rates may be below the configured sampling probabilities.

Validation (including separate validation folders), fixed TensorBoard validation
previews and inference use their existing preprocessing without augmentation.
Sampling uses PyTorch's seeded RNG; augmentation sampling with the same seed
and worker count is reproducible, and checkpoints preserve the RNG used by
augmentation. This does not guarantee deterministic GPU model computation.

Run the augmentation checks without starting a full training run:

```bash
python checks/check_frame_augmentation.py
```

## Corrected CUSTOM loss and comparing improvements

`CUSTOM` now uses **soft intersection** `sigmoid(logits) * target`, including
low-probability foreground pixels. The original normalization, smoothing and
exponent remain unchanged:

```text
p = sigmoid(logits)
CUSTOM = 1 - ((2 * sum(p * target) + 1) / (sum(p * p) + sum(target) + 1)) ** (1 / beta)
beta = 2
```

Effective configs and new checkpoints record `loss_version: custom_soft_v1`.
Old unversioned CUSTOM runs used a thresholded intersection that could push
low-confidence foreground predictions toward background. `--gain` and the
Python loss argument `threshold` are still accepted for older commands/callers,
but **have no effect on corrected CUSTOM**. Metric thresholds remain 0.5.
Auxiliary outputs still contribute fully to the optimized training loss; model
architecture and the default backbone unfreeze at epoch 300 are unchanged.

For the reviewed run, start a **new run using `--transfer`** from its best-IoU
checkpoint (epoch 16, IoU about 0.218). Keep the original **480×272** setup first
to measure the loss correction without changing resolution or input mode:

```bash
python tool/train.py \
  --data_path datasets/MOD_ONESHOT \
  --transfer work_seg/checkpoint/L_CUSTOM_2026-10-18-17-35-56-548300/checkpoint/best_iou.pth \
  --resize 480 272 \
  --batch_size 4 \
  --num_workers 6 \
  --frame_skip 3 \
  --val_split 0.2 \
  --loss CUSTOM \
  --seg_input flow \
  --stabilization homography \
  --learning_rate 0.001 \
  --epoch_training_backbone 300 \
  --backbone_layer_lr 0.1 \
  --tb_image_freq 10 \
  --tb_num_images 4 \
  --device cuda:0
```

Transfer loads weights with a fresh optimizer, scheduler, best metrics and
history. Resume restores the old optimizer/epoch and uses the corrected loss;
the session snapshot records its source loss version. Prefer transfer for
comparing the corrected loss against the old run. Absolute training loss values
from the two formulas are not directly comparable. Historical epoch losses
missing from the old run cannot be recovered; new logs begin with this code.

After that baseline, run separate experiments from the **same source checkpoint**:

- Change only `--resize 480 272` to `--resize 960 544` to test small-object detail.
  Keep batch size 4 if GPU memory permits; record any batch-size change.
- Keep the baseline resolution and change only `--seg_input flow` to
  `--seg_input flow_rgb` to test appearance cues. Transfer initializes the RGB
  weights to zero and copies the original flow weights.

Compare `best_iou.pth`, recall, precision, `val_niou_positive` and preview images
on the same validation split. Roughly 44.6% of the reviewed validation masks
are empty, so an all-background prediction can look attractive under Fa alone.
`best_FA.pth` is not the primary checkpoint for segmentation quality. Higher
resolution or RGB inputs need validation; neither guarantees improvement.

## Epoch history, sessions and TensorBoard

Each run writes these files automatically, including with `--no-tensorboard`:

```text
<run>/
  config.json                     # original effective run config
  history.jsonl                   # one JSON object per completed epoch
  history.csv                     # same fields, spreadsheet-friendly
  train.log                       # startup, epoch summaries, error tracebacks
  summary.json                    # current session status, latest epoch and bests
  checkpoint/last.pth
  checkpoint/best_iou.pth          # primary quality checkpoint
  sessions/<session_id>/
    session.json                  # config, command, runtime, resume source, samples
    config.json                   # effective config for this invocation
    summary.json                  # this session's status and best metadata
    checkpoint/epoch_XXXXXX.pth    # retained on a new best OR every --save_freq epochs
  tensorboard/<session_id>/       # optional scalar/image event files
```

All completed epochs are appended and flushed immediately. Progress bars stay
in the console; `train.log` contains epoch summaries rather than batch output.
In addition to `last.pth`, full checkpoints are retained at epochs **100, 200,
300, ...** by default (`epoch % 100 == 0`), even when no validation metric
improves or validation is skipped/disabled. For example, epoch 100 is kept at
`sessions/<session_id>/checkpoint/epoch_000100.pth`. These checkpoints include
model weights, optimizer, scheduler, scaler, RNG state and epoch metrics, and
can be passed directly to `--resume`. Validation fields are null if that epoch
did not run validation. The checkpoint path is also recorded in `train.log`.

Use `--save_freq 50` to keep every 50th epoch, or `--save_freq 0` to disable
periodic retention. This option applies to the current invocation, including
resume; omitted means 100. Saving follows the absolute epoch number, not the
number of epochs since resume. Each session has its own archive, so repeating
an epoch after resuming an older checkpoint preserves the earlier session's
copy. Periodic saving does not change the per-epoch log or `last.pth` frequency.

Best-checkpoint selection uses validation metrics:

| Checkpoint | Updated when |
| --- | --- |
| `best_iou.pth` | Global pixel IoU increases; primary segmentation checkpoint |
| `best_nIoU.pth` | Mean per-image IoU over all validation images increases |
| `best_PD.pth` | Object detection probability increases |
| `best_FA.pth` | False alarm ratio decreases |
| `last.pth` | Every epoch completes, regardless of validation |

`summary.json` records `running`, `completed`, `interrupted` or `failed`, the last
completed epoch, and each best value with its epoch and checkpoint path (paths
are relative to the run). Startup failures are also logged. An interruption in
the middle of an epoch keeps previously completed history and checkpoints;
the unfinished epoch is not recorded as complete.

Main fields in CSV/JSONL and checkpoint `epoch_metrics`:

| Fields | Meaning |
| --- | --- |
| `train_loss_total` | Actual optimized loss, including every auxiliary output and optional teacher weighting |
| `train_loss_main`, `train_loss_aux` | Unweighted supervised main loss and sum of auxiliary losses |
| `train_loss_kd` | Raw teacher distillation loss; zero without a teacher |
| `val_loss` | Main prediction loss only; not directly comparable to the sum of training outputs |
| `train_iou`, `val_iou`, `*_precision`, `*_recall` | Confusion counts aggregated over evaluated pixels, at threshold 0.5 |
| `*_target_fg_ratio`, `*_pred_fg_ratio` | Foreground pixels / all evaluated pixels, in [0, 1] |
| `val_niou`, `val_niou_positive` | Mean per-image IoU over all images / only images with foreground in resized labels |
| `val_positive_samples`, `val_pd`, `val_fa` | Foreground image count, object detection probability and false alarm pixel ratio |
| `train_samples`, `val_samples` | Actually processed samples; training drops incomplete batches |
| `lr_head`, `lr_backbone` | Sample-weighted mean LR used by successful optimizer updates, before scheduler steps |
| `lr_*_first`, `lr_*_last` | First/last rates actually used in the epoch; backbone is zero while frozen |
| `backbone_training` | Whether the backbone is unfrozen |
| `train_seconds`, `val_seconds`, `epoch_seconds` | Train/validation wall time and their sum, excluding checkpoint writes |
| `amp_scale`, `optimizer_steps`, `optimizer_skipped_steps` | Final scaler value and successful/overflow-skipped optimizer updates |
| `gpu_peak_memory_mb` | Peak allocated PyTorch GPU memory during train+validation, in MiB; null on CPU |

Loss means are weighted by sample count, including the last validation batch.
Empty-target/empty-prediction images contribute zero to all-image nIoU, as
before. `val_niou_positive` is null if no evaluated image has foreground. All
validation fields are **null** on epochs skipped by `--eval_freq`, or when
`--val_split 0` disables validation. All-skipped optimizer epochs report zero
actual LR. With a teacher, total loss is
`alpha * (main + aux) + (1 - alpha) * kd`.

Every invocation, including resume, gets a new `session_id`, configuration
snapshot, command line and Python/PyTorch/CUDA versions. Resume appends rows
without deleting earlier sessions; repeated epoch numbers are distinguished by
session ID. New checkpoint metadata uses plain dictionaries, lists and native
Python values, compatible with PyTorch 2.3 safe loading. Best epochs absent
from old checkpoints are recovered only when a matching best file is available;
unknown epochs/paths remain null rather than being guessed.

Best checkpoint snapshots preserve their epoch across later improvements and
resume from earlier epochs. They share disk storage with current checkpoint
aliases using hard links when supported; otherwise files are copied. Retaining
earlier best epochs uses additional space. The standard `checkpoint/best_iou.pth`
tracks the active resume branch, and session summaries retain earlier references.

Install and start TensorBoard from the active training environment:

```bash
python -m pip install tensorboard
tensorboard --logdir work_seg/checkpoint --port 6006
```

Open **http://localhost:6006**. Use **Scalars** for `Epoch/train_loss_total`,
`Epoch/train_loss_main`, `Epoch/train_loss_aux`, validation metrics and learning
rates. Use **Images** for each fixed sample's left-to-right comparison:
**current RGB / ground-truth mask / predicted binary mask / probability**.

The default four samples prefer three foreground masks and one empty mask,
spread across sequences; unavailable categories are filled with other valid
samples. The chosen frame pairs are saved in the session and checkpoint and
kept on resume when still present. `--tb_num_images 4` changes the count.
Images are logged on the first validation of each session, on validation epochs
divisible by `--tb_image_freq 10`, and whenever IoU improves, at most once per
epoch. Images/logits come directly from the validation loop, with **no extra
forward**. Only TensorBoard event files contain images; no PNG/JPG files are
exported. Session-specific event directories avoid overlapping curves if an
older epoch is resumed. Missing TensorBoard leaves the persistent history intact.

Run the logging/loss checks alongside the existing training checks:

```bash
python checks/check_training_logging.py
python checks/check_frame_training.py
python checks/check_checkpoint_compat.py
```

## Launch scripts

Activate your PyTorch environment first, or set `PYTHON_BIN=/path/to/python`.
The launchers can run from any working directory; paths supplied by you remain
relative to that working directory. All Python CLI options are supported.

```bash
# Train either head mode.
bash scripts/train.sh --data_path /data/train --seg_input flow
bash scripts/train.sh --data_path /data/train --seg_input flow_rgb

# Initialize RGB from a flow checkpoint, with a fresh optimizer.
bash scripts/train.sh --data_path /data/train --seg_input flow_rgb \
  --transfer /results/flow/checkpoint/last.pth

# Resume: the head mode is restored from the checkpoint.
bash scripts/train.sh --data_path /data/train \
  --resume /results/rgb/checkpoint/last.pth

# Past image is already warped by the external system.
bash scripts/infer.sh --checkpoint /results/rgb/checkpoint/last.pth \
  --current /data/current.png --past /data/past_warped.png \
  --output /results/mask.png --iters_s8 10
```

Training uses the Python trainer's existing defaults, including training-time
homography stabilization. Outputs default to `<repository>/work_seg/checkpoint`;
change this with `WORK_DIR` or `--work_dir`. If `--seg_input`/`SEG_INPUT` is unset,
new training runs use `NeuFlow/config.py`, and resume uses the checkpoint.
Inference disables internal warp by default and restores the mode, resolution
and iteration settings from the checkpoint unless explicitly overridden.

The scripts also support environment configuration:

```bash
DATA_PATH=/data/train SEG_INPUT=flow_rgb bash scripts/train.sh
CHECKPOINT=/results/rgb/checkpoint/last.pth \
  CURRENT_FRAME=/data/current.png PAST_FRAME=/data/past_warped.png \
  OUTPUT_MASK=/results/mask.png bash scripts/infer.sh
```

CLI values override environment values for the same option. Use `--help` for
launcher configuration, `--python-help` for the complete Python options, or
`--dry-run` to print the exact command without starting training/inference.

NeuFlow is the optical-flow backbone of the segmentation model. Training only
needs two input frames and the ground-truth mask of the current frame; **no
optical-flow ground truth or separate optical-flow loss is required**.

```text
current frame ---------------------------------------> NeuFlow S16/S8
past frame -> homography -> warp to current frame ---->       |
                                                       final flow
                                                            |
                                                     segmentation head
                                                            |
                                               mask loss vs current-frame mask
```

The training objective is the segmentation loss, including the segmentation
head's auxiliary mask outputs. Once the backbone is unfrozen, backpropagation
through the head and flow normalization updates NeuFlow as well. The MOD model
keeps the accumulated flow differentiable between iterations, so mask loss
reaches every S16/S8 flow update. All refinement steps and training predictions
are retained; S16/S8 use one/eight iterations by default. This frame/mask training
setup needs only image and mask folders, with no teacher/distillation model.

```bash
python tool/train.py \
  --image_dir /data/train/images \
  --mask_dir /data/train/masks \
  --val_image_dir /data/val/images \
  --val_mask_dir /data/val/masks \
  --frame_skip 3 \
  --stabilization homography \
  --iters_s16 1 \
  --iters_s8 8 \
  --resize 960 544 \
  --epoch_training_backbone 300 \
  --backbone_layer_lr 0.1
```

Each sample is **(current frame `t`, past frame `t - frame_skip`, mask at `t`)**.
`frame_skip=1` uses adjacent frames. The gap is measured in positions in the
naturally sorted file list (`frame_2.jpg` precedes `frame_10.jpg`), not seconds or
the difference between numeric filenames. The first `frame_skip` frames in each
sequence have no earlier partner and are omitted as targets.

Masks may cover only some frames. Pairs are constructed from the complete image
list, then kept only when the current frame has a matching mask. The past frame
does not need a mask; unmasked images still count toward `frame_skip`. For images
1–5 with masks only for 3 and 5, `frame_skip=1` produces `(3, 2, mask 3)` and
`(5, 4, mask 5)`. Train/validation splitting happens before pairing and mask
filtering, so pairs never cross split boundaries. This also applies to separate
validation folders. A selected train or validation set with no valid masked
pairs raises an error; validation is not silently disabled.

Images and ground-truth masks must have the same relative path and filename stem;
their extensions may differ. Use one folder per video. Pairs never cross folders.
Both a flat folder and this nested layout are supported:

```text
images/
  video_1/frame_001.jpg
  video_1/frame_002.jpg
  video_2/frame_001.jpg
masks/
  video_1/frame_001.png
  video_1/frame_002.png
  video_2/frame_001.png
```

Masks must be **PNG or BMP**. JPEG, WebP, TIFF and other image formats in the mask
folder are rejected; the reader also checks the decoded format to catch JPEG
files renamed to `.png` or `.bmp`. Export masks from the original annotations:
converting a compressed JPEG mask to PNG does not recover its original labels.
Input images can still be JPEG or any other previously supported image format.

Masks are binary: zero is background, positive values are foreground, including
both 0/1 and 0/255 encodings. PNG palette masks use their class IDs. A mask can
have a different resolution from its image, but must describe the same full
frame (without cropping or padding). The two input frames must have matching
original dimensions. Images convert to RGB in the range 0–255, as expected by NeuFlow.

`--resize WIDTH HEIGHT` resizes both frames with bilinear interpolation and
the mask with nearest-neighbor interpolation. `--img_size` is an alias for
the same option. Each requested dimension must be at least 32; multiples of
16 are not required for training or PyTorch inference.

Without either option, frames keep their original resolution and only the
mask resizes to match the current frame. For example, `MOD_ONESHOT` images
are 1920×1080 and its masks are 960×544:

```bash
# Frames and masks become 960×544.
python tool/train.py --data_path datasets/MOD_ONESHOT --resize 960 544

# Frames stay 1920×1080; masks resize to 1920×1080.
python tool/train.py --data_path datasets/MOD_ONESHOT
```

ModelBuilder replicates the right/bottom edge to reach multiples of 16
internally, then crops every prediction (including training auxiliary outputs)
back to the requested or native resolution before loss/metrics. For 1920×1080,
computation uses 1920×1088 and predictions remain 1920×1080. PD/FA normalizes
false alarms by the actual number of evaluated pixels. Samples within each
batch must share a resolution; use `--resize` if native resolutions differ.

## Stabilizing the past frame

New training runs default to `--stabilization homography`. Training, validation
and inference use the same preprocessing in `data_utils/frame_pair.py`. Only the
past frame is warped into the current frame's coordinates; the current frame
and ground-truth mask receive their usual resize and no geometric warp. The
stabilizer uses only the two images, never ground-truth masks. It runs on CPU
before the model; the segmentation loss still trains the unfrozen NeuFlow.

The implementation uses OpenCV SIFT (up to 2,000 features), mutual descriptor
matches with a 0.75 Lowe ratio, and RANSAC homography from past to current.
Estimation uses a maximum long side of 960 pixels; the transform is converted
back to model-input coordinates before bilinear warping. OpenCV with SIFT is
required (for example `opencv-python-headless >= 4.4`).

RANSAC uses a 3-pixel reprojection threshold at the estimation resolution,
2,000 maximum iterations and 0.995 confidence. Acceptance requires at least 12
inliers, an inlier ratio of 50%, an inlier hull covering at least 5% of each
image, and at least 50% valid overlap. Singular/reflected transforms and
transforms crossing the projective horizon are rejected. Missing warped pixels
are filled from the current frame; those regions have no inter-frame motion
signal. Failed alignment falls back to the unwarped past frame. Training warns
once per failure reason in each dataset worker; inference prints its status.

`--stabilization none` reproduces the previous RGB/resize preprocessing. The
selected mode is saved in checkpoint arguments. Resume and inference inherit
the saved mode unless explicitly overridden; old checkpoints without this
field default to `none`. To enable alignment for an old checkpoint, explicitly
pass `--stabilization homography`. Direct `FrameMaskDataset` callers also default
to `none` for compatibility; the trainer passes its resolved mode explicitly.

A single homography describes a planar scene or predominantly rotational camera
motion. Parallax from different depths, sparse texture, or moving objects
dominating the matches can reduce alignment quality. The checks reject obvious
failures but cannot guarantee that every accepted transform follows the
background. Validate segmentation quality on your camera data.

Without separate validation folders, `--val_split 0.2` reserves the last 20% of
frames in each sequence for validation **before** forming pairs. Train and
validation therefore share no frames. Each selected sequence portion must have
more than `frame_skip` frames to produce samples. `--val_split 0` disables
validation. Separate videos for validation are preferable when available.
`--data_path /data/train` is shorthand for `/data/train/images` and `/data/train/masks`.

Epochs **1–299 freeze all backbone weights and BatchNorm statistics**. Epoch
**300** starts training the backbone at `0.1 ×` the head's current learning rate.
The optimizer and scheduler continue without restarting. `--epoch_training_backbone 1`
starts backbone training immediately. The default pretrained backbone is the
repository's `neuflow_sintel.pth`; use `--backbone_op_pretrained PATH` to change it,
or `--backbone_op_pretrained ''` for random initialization.

The current trainer requires a training batch size of at least two and drops
incomplete training batches; validation keeps all samples. For a CPU smoke run,
use e.g. `--device cpu --img_size 32 32
--batch_size 2 --num_workers 0 --epochs 2 --warm_up_epochs 0 --no-amp --no-tensorboard`.

Checkpoints are written under `--work_dir`, with `checkpoint/last.pth` saved each
epoch, even when validation is disabled. Resume with the same data, scheduler
and iteration settings and `--resume /path/to/checkpoint/last.pth`. Checkpoints
save `img_size` as the selected `[WIDTH, HEIGHT]`, or `null` for native frames,
and save `iters_s16`/`iters_s8` in their arguments. Training uses the current
CLI resize choice, including on resume; repeat `--resize` to keep a resized run.
Older checkpoints without the iteration fields still load with the CLI defaults
of 1/8. Checkpoints from the earlier
single-parameter-group optimizer should instead be loaded with `--transfer` to
initialize weights and start a new optimizer.

## Two-image segmentation inference

```bash
python tool/infer.py \
  --checkpoint /path/to/checkpoint/last.pth \
  --current /data/frame_010.png \
  --past /data/frame_007.png \
  --output /results/frame_010_mask.png \
  --stabilization homography \
  --iters_s8 10
```

Supply a full **ModelBuilder segmentation checkpoint**. The two original images
must have matching dimensions. The command stabilizes the past image, runs the
model, resizes logits back to the current image's original size, applies sigmoid
and `--threshold` (default 0.5), and writes a single-channel PNG containing 0/255.
The output remains in current-frame coordinates. Missing output directories are
created; either input image and the checkpoint are protected from overwrite.

`--resize WIDTH HEIGHT` (alias `--img_size`), `--iters_s16` and `--iters_s8`
override checkpoint settings. A checkpoint with `img_size: null` keeps the
original frame size; an explicit resize overrides it. Legacy checkpoints with
no size setting retain the 960×544 fallback. Missing iteration settings default
to S16=1 and S8=8. Stabilization follows the checkpoint rules above. `--device`
defaults to CUDA when available, otherwise CPU; `--no-amp` disables CUDA AMP.
The script prints alignment success/fallback and the chosen model input size.

## Inference with a different number of iterations

`--iters_s16` and `--iters_s8` must be positive integers. They control training
and validation, and the teacher when one is configured. Inference chooses them
independently through `ModelBuilder.forward`; all iterations share weights, so
a checkpoint trained with eight S8 iterations can run ten without changing its
state dictionary:

```python
import torch
from NeuFlow.model_builder import ModelBuilder
from tool.checkpoints import load_weights
from data_utils.frame_pair import load_frame_pair

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = ModelBuilder().to(device)
load_weights(model, '/path/to/checkpoint/last.pth')
model.eval()
current, past, alignment, original_size = load_frame_pair(
    '/data/current.png', '/data/past.png', img_size=(960, 544),
    stabilization='homography',  # Use the mode selected during training.
)
current, past = current.unsqueeze(0).to(device), past.unsqueeze(0).to(device)
with torch.inference_mode():
    logits = model(current, past, iters_s16=1, iters_s8=10)
    mask = torch.sigmoid(logits) > 0.5
```

Compare validation quality and runtime at 8 and 10 before choosing an inference
setting; additional refinement does not guarantee better masks. Full gradient
propagation can increase training memory after unfreezing the backbone. Direct
calls to `NeuFlow.forward` retain the legacy `detach_flow_between_iters=True`
default; `ModelBuilder` explicitly disables it for MOD. This API example returns
a mask at model-input resolution; `tool/infer.py` restores the original size.

Training accepts only image/mask datasets. The `--dataset` option and support for
precomputed-flow datasets have been removed; NeuFlow computes flow from each
image pair. The `--seg_input flow` and `--seg_input flow_rgb` head modes remain
available with image/mask datasets.

Run the dataset, freeze/LR, metric, and real-model resume checks with:

```bash
python checks/check_frame_training.py
python checks/check_stabilization.py
```
# Select the segmentation head input

Set `seg_input = "flow"` or `seg_input = "flow_rgb"` in `NeuFlow/config.py`.
An explicit `--seg_input` overrides that default for a new training run:

```bash
python tool/train.py --data_path /data/train --seg_input flow
python tool/train.py --data_path /data/train --seg_input flow_rgb
```

`flow` supplies normalized `[u, v]` to the existing head. `flow_rgb` supplies
`[u, v, R, G, B]`, using the **current frame**, RGB order, divided by 255 and
converted to the flow dtype. The stem convolution changes from 2→6 to 5→6
channels; max-pooling still sees only the first two flow channels. The combined
stem output remains eight channels. There is no additional image encoder.

The effective selection is saved in `config.json` and checkpoint `args`.
Resume and inference restore it before constructing the model; a conflicting
CLI selection is rejected. A legacy checkpoint with no `seg_input` means
`flow`, even if the default in `NeuFlow/config.py` has since changed.
`ModelBuilder(seg_input=None)` uses the Python config default; explicit
`ModelBuilder(seg_input="flow_rgb")` selects RGB directly.

Use **transfer**, with a fresh optimizer, to extend a trained flow model:

```bash
python tool/train.py --data_path /data/train --seg_input flow_rgb \
  --transfer /results/flow/checkpoint/last.pth
```

The six original flow filters are copied and the three new input-channel
weights are zeroed. Initial predictions match the old model, while RGB weights
receive gradients during training. Other weights retain their values. Same-mode
transfer is supported; conversion from a trained RGB head back to flow is
rejected rather than silently discarding learned RGB inputs. Changing modes
with `--resume` is rejected.

When an external system already warps the past image into current-frame
coordinates, explicitly disable the built-in warp:

```bash
python tool/infer.py --checkpoint /results/rgb/checkpoint/last.pth \
  --current /data/current.png --past /data/past_already_warped.png \
  --output /results/mask.png --stabilization none
```

The head mode is restored from this checkpoint automatically. Both model inputs
remain RGB images in `[0,255]`, including for the RGB head; do not divide by
255 before calling `ModelBuilder`.

## Export the full segmentation model

Install `tool/requirements-export.txt` in the PyTorch environment. TensorRT is
not needed to prepare and validate ONNX on a CPU workstation.

```bash
python -m pip install -r tool/requirements-export.txt
python tool/export_seg.py --checkpoint /results/rgb/checkpoint/last.pth \
  --output /results/rgb.onnx --img_size 960 544 --iters_s8 8
```

This exports **NeuFlow plus segmentation**, with fixed batch one, FP32 NCHW
`current`/`past` inputs and FP32 `logits` output. Normalization and RGB
concatenation are inside the graph; camera warp, image resize and output
thresholding remain outside. The old root `pt2onnx.py` exports only optical
flow and should not be used for these segmentation checkpoints.

Static export still requires both dimensions to be at least 32 and divisible
by 16. When the checkpoint uses native-size frames (`img_size: null`), provide
an explicit `--img_size WIDTH HEIGHT` for export.

Export works on a copy, replacing each zero-hidden-channel SE block with its
exact `x * 0.5` equivalent. It checks the ONNX model and compares logits against
PyTorch on two deterministic frame pairs (`atol=rtol=1e-3`). A `.json` sidecar
records checks/configuration and the ONNX SHA256; `.validation.npz` stores the
inputs and reference logits for TensorRT verification. Export fails on a
numerical mismatch. These checks validate execution, not detection accuracy.

## Compare TensorRT FP16 latency

Prepare matching baselines and RGB experiments on the workstation:

```bash
python tool/benchmark_seg.py --phase prepare --output_dir /results/head_benchmark \
  --checkpoint /results/flow/checkpoint/last.pth --img_size 960 544 --iters_s8 8 10
```

The starting checkpoint must be a full **flow-only** model. If omitted, the
tool uses `neuflow_sintel.pth` and a seeded, untrained head. All common weights
are identical between models. For this timing experiment only, new RGB weights
use deterministic nonzero values, preventing a trivially unused RGB path.
The benchmark weights are not a trained detection model.

Copy the complete artifact directory and this repository to the measuring
machine, then run in its PyTorch/CUDA/TensorRT >= 8.6 environment:

```bash
python tool/benchmark_seg.py --phase run --output_dir /results/head_benchmark
```

On Jetson use the CUDA/TensorRT packages supplied by JetPack. Engines are built
on the measuring GPU, with FP16 enabled, TF32 disabled, FP32 I/O, and equal
workspace limits. Some layers may retain FP32 for supported fallback behavior.
Engines built on RTX are not copied to Orin.

Before timing, TensorRT logits must pass the saved PyTorch references with
`atol=rtol=0.01` and binary-mask agreement >= 99.5%. Tolerances are configurable
and recorded. A failed validation prevents a latency conclusion. Any additional
export/precision optimization must be applied to both models and revalidated.

Defaults: batch 1, 960×544, S16=1, S8=8 and 10, five rounds with at least 1,000
samples per mode/stage, and at least three seconds of warm-up before each stage.
Model order alternates between rounds. Device-resident measurements use CUDA
events around the entire graph. A second stage measures host-observed latency
including H2D/D2H using preallocated pinned buffers. Both include graph RGB
normalization/concatenation and exclude file I/O, engine builds and external
warp. There is no CUDA graph or concurrent inference.

`timings.csv` contains individual samples. `report.json` contains median, p95,
absolute/relative increases, round consistency, numerical validation, hardware
and temperature snapshots. A pass requires **both median and p95 <= +5%** for
GPU inference and the transfer-inclusive path, in aggregate and in every round.
A passing aggregate with failing individual rounds is `inconclusive`.

Memory fields distinguish TRT's context workspace, PyTorch's peak allocation
(which excludes TRT), and sampled global device use. The latter is an observed
high-water mark, not an exact trace of transient allocation peaks. Run on an
otherwise idle device; global memory and temperature can include other work.

Without a GPU, use the arithmetic-only report:

```bash
python tool/benchmark_seg.py --phase estimate --output_dir /results/head_estimate
```

At 960×544 the modified convolution adds 162 parameters and 21,150,720 MACs.
Unfused FP16 RGB/concatenation intermediates are approximately 2.99/4.98 MiB;
TensorRT may fuse them. These figures do **not** predict milliseconds or prove
the 5% limit. A failed GPU/export preflight writes `status: not_measured`, with
no latency pass. RTX 2060 SUPER measurements are reference measurements only,
not estimates of Jetson AGX Orin performance.

Run the behavior checks, including actual ONNX validation when dependencies are
installed:

```bash
python checks/check_frame_training.py
python checks/check_stabilization.py
python checks/check_seg_input.py
```
