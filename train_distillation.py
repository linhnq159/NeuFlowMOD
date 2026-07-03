"""
Main script to train NeuFlowV2 using distillation from SEA-RAFT.
"""
import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import os

# Add segmentation for postprocessing
repo_SEARAFT = Path("/data/linhnq11/MOD/SEA-RAFT")
sys.path.append(str(repo_SEARAFT))

# Import your models and utils
from NeuFlow.neuflow import NeuFlow
from core.raft import RAFT
from core.utils.utils import load_ckpt
# from data.dataset import OpticalFlowDataset

from distillation_trainer import DistillationTrainer
from distillation_config import DISTILLATION_CONFIG

from config.parser import parse_args

import argparse
import json

from data_utils.datasets import build_train_dataset

def get_args_parser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--checkpoint_dir', default=None, type=str)
    parser.add_argument('--dataset_dir', default=None, type=str)
    parser.add_argument('--stage', default='things', type=str)
    parser.add_argument('--val_dataset', default=['things', 'sintel'], type=str, nargs='+')

    # training
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--val_freq', default=1000, type=int)
    parser.add_argument('--num_steps', default=1000000, type=int)

    parser.add_argument('--max_flow', default=400, type=int)

    # resume pretrained model or resume training
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--strict_resume', action='store_true')

    # distributed training
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--distributed', action='store_true')

    # use fp16
    parser.add_argument('--use_fp16', action='store_true')

    return parser

def json_to_args(json_path):
    # return a argparse.Namespace object
    with open(json_path, 'r') as f:
        data = json.load(f)
    args = argparse.Namespace()
    args_dict = args.__dict__
    for key, value in data.items():
        args_dict[key] = value
    return args

def load_models(args, config):
    """Load student and teacher models."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # parser_SEARAFT = argparse.ArgumentParser()
    # parser_SEARAFT.add_argument('--cfg', help='experiment configure file name', default= config['searaft_config'], type=str)
    # parser_SEARAFT.add_argument('--model', help='checkpoint path', default= config['searaft_checkpoint'], type=str)
    # args_SEARAFT = parse_args(parser_SEARAFT)

    args_SEARAFT = argparse.Namespace(
        cfg = config['searaft_config'],
        model = config['searaft_checkpoint']
    )

    entry = args_SEARAFT
    json_path = entry.cfg
    print("json_path : ", json_path)
    args_SEARAFT = json_to_args(json_path)
    args_dict = args_SEARAFT.__dict__
    for index, (key, value) in enumerate(vars(entry).items()):
        args_dict[key] = value

    # Load NeuFlowV2 (student)
    student = NeuFlow()
    if config['neuflow_checkpoint']:
        if args.resume  : 
            # print("config['neuflow_checkpoint'] : ", config['neuflow_checkpoint'])
            checkpoint = torch.load(args.resume, map_location=device)
        else:
            checkpoint = torch.load(config['neuflow_checkpoint'], map_location=device)
        student.load_state_dict(checkpoint['model'], strict = True)
    

    #Load SEA-RAFT (teacher)
    teacher = RAFT(args_SEARAFT)
    if config['searaft_checkpoint']:
        # checkpoint = torch.load(config['searaft_checkpoint'], map_location=device)
        # teacher.load_state_dict(checkpoint)
        load_ckpt(teacher, config['searaft_checkpoint'])
        teacher = teacher.to(device)
    
    return student, teacher


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


def main(args):
    """Main training function."""
    config = DISTILLATION_CONFIG
    
    print("=" * 80)
    print("NeuFlowV2 Distillation Training (Teacher: SEA-RAFT)")
    print("=" * 80)
    print(f"Config: {config}")
    
    # Load models
    student, teacher = load_models(args, config)
    if student is None or teacher is None:
        print("ERROR: Failed to load models. Please adjust model loading in load_models()")
        sys.exit(1)
    
    # # Create dataloaders
    # train_loader, val_loader = create_dataloaders(config)
    # if train_loader is None or val_loader is None:
    #     print("ERROR: Failed to create dataloaders. Please adjust in create_dataloaders()")
    #     sys.exit(1)

    ### Create train_loader 
    train_dataset, val_dataset = build_train_dataset(args.stage)

    print('Number of training images:', len(train_dataset))
    # print('Number of validation images:', len(val_dataset))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=torch.cuda.device_count(),
            rank=args.local_rank)
    else:
        train_sampler = None

    shuffle = False if args.distributed else True
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=shuffle, num_workers=args.num_workers,
                                               pin_memory=True, drop_last=True,
                                               sampler=train_sampler)

    
    # Initialize trainer
    trainer = DistillationTrainer(
        student_model=student,
        teacher_model=teacher,
        train_loader=train_loader,
        val_dataset=val_dataset,
        train_sampler = train_sampler,
        config=config,
        args = args,
        use_fp16 = args.use_fp16
    )
    
    # Start training
    print("\nStarting distillation training...")
    trainer.train(
        num_epochs=config['num_epochs'],
        checkpoint_dir=args.checkpoint_dir
    )
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)


if __name__ == '__main__':
    parser = get_args_parser()
    print("parser : ", parser)
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    print("args : ", type(args))
    print("args : ", args)

    main(args)