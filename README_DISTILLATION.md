# NeuFlowV2 Distillation Training

This module implements knowledge distillation for NeuFlowV2 using SEA-RAFT as a teacher model.

## Overview

- **Student Model**: NeuFlowV2 (learns to predict optical flow)
- **Teacher Model**: SEA-RAFT (provides soft targets for guidance)
- **Loss Function**: Combination of task loss and KL divergence between predictions

## Files

- `distillation_loss.py`: Distillation loss and task loss implementations
- `distillation_trainer.py`: Main trainer class
- `distillation_config.py`: Training configuration
- `train_distillation.py`: Entry point for training

## Quick Start

1. **Configure hyperparameters** in `distillation_config.py`:
   - `alpha`: Weight for distillation vs task loss (0.7 = 70% distillation)
   - `temperature`: Controls softness of probability distributions
   - `learning_rate`: Learning rate for student model

2. **Prepare model checkpoints**:
   - Place NeuFlowV2 checkpoint at path specified in config
   - Place SEA-RAFT checkpoint at path specified in config

3. **Run training**:
   ```bash
   python train_distillation.py
   ```

## Configuration

Key parameters in `distillation_config.py`:

```python
DISTILLATION_CONFIG = {
    'alpha': 0.7,              # Distillation loss weight
    'temperature': 4.0,        # Temperature for soft targets
    'learning_rate': 1e-4,     # Student learning rate
    'batch_size': 32,
    'num_epochs': 400,
    'lr_milestones': [100, 200, 300],
}
```

## Loss Function

Total Loss = (1 - α) × TaskLoss + α × DistillationLoss

- **TaskLoss**: Charbonnier loss between student prediction and ground truth
- **DistillationLoss**: KL divergence between student and teacher predictions

## Output

Checkpoints are saved in `./checkpoints/distillation/`:
- `best_model.pth`: Best model based on validation loss
- `model_epoch_N.pth`: Checkpoints at regular intervals

## Tips for Better Results

1. Use a well-trained SEA-RAFT model as teacher
2. Adjust `alpha` based on your needs:
   - Higher α: More emphasis on teacher guidance
   - Lower α: More emphasis on ground truth
3. Start with temperature=4.0, adjust if needed
4. Monitor both task loss and distillation loss during training


## Exp:
python train_distillation.py --checkpoint_dir experiment/16062026_resume/ --stage custom --val_dataset U3_giam_chan --batch_size 32 --num_workers 4 --lr 1e-4 --val_freq 1000 --resume experiment/16062026/step_002000.pth --strict_resume --lr 0.0001


CUDA_VISIBLE_DEVICES=1 python train_distillation.py --checkpoint_dir experiment/07072026_resume_1/ --stage custom --val_dataset U3_giam_chan --batch_size 16 --num_workers 4 --lr 1e-4 --val_freq 2 --strict_resume --lr 0.0001 --resume neuflow_mixed.pth