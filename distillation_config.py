"""Configuration for NeuFlowV2 distillation training with SEA-RAFT as teacher."""

DISTILLATION_CONFIG = {
    # Model paths
    'neuflow_checkpoint': './checkpoints/neuflowv2.pth',
    'searaft_checkpoint': '../SEA-RAFT/checkpoints/searaft.pth',
    
    # Loss weights
    'alpha': 0.7,  # Weight for distillation loss (1-alpha for task loss)
    'temperature': 4.0,  # Temperature for softening distributions
    'epsilon': 0.01,  # Epsilon for Charbonnier loss
    
    # Optimizer
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'grad_clip': 1.0,
    
    # Learning rate scheduler
    'lr_milestones': [100, 200, 300],
    'lr_gamma': 0.1,
    
    # Training
    'batch_size': 32,
    'num_epochs': 400,
    'num_workers': 8,
    'log_interval': 10,
    'save_interval': 10,
    
    # Data
    'train_data_root': './data/FlyingChairs',
    'val_data_root': './data/validation',
    'image_size': (384, 512),  # (height, width)
}
