"""
Main script to train NeuFlowV2 using distillation from SEA-RAFT.
"""
import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader

# Import your models and utils
# from neuflow import NeuFlowV2
# from sea_raft import SEARAFT
# from data.dataset import OpticalFlowDataset

from distillation_trainer import DistillationTrainer
from distillation_config import DISTILLATION_CONFIG


def load_models(config):
    """Load student and teacher models."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load NeuFlowV2 (student)
    # from neuflow import NeuFlowV2
    # student = NeuFlowV2()
    # if config['neuflow_checkpoint']:
    #     checkpoint = torch.load(config['neuflow_checkpoint'], map_location=device)
    #     student.load_state_dict(checkpoint)
    
    # Load SEA-RAFT (teacher)
    # from sea_raft import SEARAFT
    # teacher = SEARAFT()
    # if config['searaft_checkpoint']:
    #     checkpoint = torch.load(config['searaft_checkpoint'], map_location=device)
    #     teacher.load_state_dict(checkpoint)
    
    # TODO: Uncomment above and adjust imports based on your project structure
    print("Models loading - adjust imports in this function")
    return None, None


def create_dataloaders(config):
    """Create training and validation dataloaders."""
    # from data.dataset import OpticalFlowDataset
    
    # train_dataset = OpticalFlowDataset(
    #     root=config['train_data_root'],
    #     image_size=config['image_size'],
    #     split='train'
    # )
    
    # val_dataset = OpticalFlowDataset(
    #     root=config['val_data_root'],
    #     image_size=config['image_size'],
    #     split='val'
    # )
    
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=config['batch_size'],
    #     shuffle=True,
    #     num_workers=config['num_workers']
    # )
    
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=config['batch_size'],
    #     shuffle=False,
    #     num_workers=config['num_workers']
    # )
    
    # TODO: Uncomment above and adjust based on your dataset structure
    print("Dataloaders creation - adjust imports in this function")
    return None, None


def main():
    """Main training function."""
    config = DISTILLATION_CONFIG
    
    print("=" * 80)
    print("NeuFlowV2 Distillation Training (Teacher: SEA-RAFT)")
    print("=" * 80)
    print(f"Config: {config}")
    
    # Load models
    student, teacher = load_models(config)
    if student is None or teacher is None:
        print("ERROR: Failed to load models. Please adjust model loading in load_models()")
        sys.exit(1)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    if train_loader is None or val_loader is None:
        print("ERROR: Failed to create dataloaders. Please adjust in create_dataloaders()")
        sys.exit(1)
    
    # Initialize trainer
    trainer = DistillationTrainer(
        student_model=student,
        teacher_model=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    # Start training
    print("\nStarting distillation training...")
    trainer.train(
        num_epochs=config['num_epochs'],
        checkpoint_dir='./checkpoints/distillation'
    )
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
