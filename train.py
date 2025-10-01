"""
Main training script for video classification with 3D CNN.
Supports multiple model architectures and comprehensive training pipeline.
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import random
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from models import create_model, get_model_info
from data import create_mock_data_loaders, create_data_loaders
from training import VideoTrainer, train_model
from configs import Config, create_config_for_model, load_config, setup_configs


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train 3D CNN for video classification')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='simple_3dcnn',
                       choices=['simple_3dcnn', 'i3d', 'slowfast', 'x3d'],
                       help='Model architecture')
    parser.add_argument('--num_classes', type=int, default=10,
                       help='Number of classes')
    
    # Data arguments
    parser.add_argument('--num_frames', type=int, default=16,
                       help='Number of frames per video')
    parser.add_argument('--resize', type=int, nargs=2, default=[112, 112],
                       help='Resize dimensions (height width)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'none'],
                       help='Learning rate scheduler')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='mock',
                       choices=['mock'],
                       help='Dataset type')
    parser.add_argument('--num_train', type=int, default=800,
                       help='Number of training samples (for mock dataset)')
    parser.add_argument('--num_val', type=int, default=200,
                       help='Number of validation samples (for mock dataset)')
    
    # Other arguments
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--project_name', type=str, default='video-classification',
                       help='Project name for wandb')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        print(f"Loaded configuration from {args.config}")
    else:
        config = create_config_for_model(args.model)
        # Override with command line arguments
        config.data.num_frames = args.num_frames
        config.data.resize = tuple(args.resize)
        config.data.batch_size = args.batch_size
        config.data.num_workers = args.num_workers
        config.training.epochs = args.epochs
        config.training.learning_rate = args.lr
        config.training.weight_decay = args.weight_decay
        config.training.scheduler = args.scheduler
        config.training.save_dir = args.save_dir
        config.logging.use_wandb = args.use_wandb
        config.logging.project_name = args.project_name
        config.model.num_classes = args.num_classes
        config.device = str(device)
        config.seed = args.seed
    
    print(f"Configuration: {config.to_dict()}")
    
    # Create model
    print(f"Creating {args.model} model...")
    model = create_model(args.model, num_classes=args.num_classes)
    
    # Print model info
    model_info = get_model_info(model)
    print(f"Model parameters: {model_info['total_parameters']:,}")
    print(f"Model size: {model_info['model_size_mb']:.2f} MB")
    
    # Create data loaders
    print("Creating data loaders...")
    
    if args.dataset == 'mock':
        train_loader, val_loader = create_mock_data_loaders(
            num_train=args.num_train,
            num_val=args.num_val,
            num_classes=args.num_classes,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_frames=args.num_frames,
            resize=tuple(args.resize)
        )
        print(f"Created mock dataset: {args.num_train} train, {args.num_val} val samples")
    else:
        raise ValueError(f"Dataset type '{args.dataset}' not supported")
    
    # Create trainer
    print("Initializing trainer...")
    trainer = VideoTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=args.num_classes,
        device=device,
        config=config.to_dict()
    )
    
    # Start training
    print("Starting training...")
    history = trainer.train(
        epochs=args.epochs,
        save_dir=args.save_dir,
        save_best=True,
        patience=config.training.patience
    )
    
    # Final evaluation
    print("Running final evaluation...")
    results = trainer.evaluate()
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print("="*50)
    print(f"Final validation accuracy: {results.get('accuracy', 0):.4f}")
    print(f"Final validation loss: {results.get('loss', 0):.4f}")
    
    # Save final results
    results_path = Path(args.save_dir) / 'final_results.json'
    import json
    with open(results_path, 'w') as f:
        json.dump({
            'model': args.model,
            'num_classes': args.num_classes,
            'final_accuracy': results.get('accuracy', 0),
            'final_loss': results.get('loss', 0),
            'history': history,
            'config': config.to_dict()
        }, f, indent=2)
    
    print(f"Results saved to {results_path}")
    
    # Plot training history
    plot_path = Path(args.save_dir) / 'training_history.png'
    trainer.plot_training_history(save_path=str(plot_path))
    print(f"Training plots saved to {plot_path}")
    
    # Plot confusion matrix
    cm_path = Path(args.save_dir) / 'confusion_matrix.png'
    trainer.plot_confusion_matrix(save_path=str(cm_path))
    print(f"Confusion matrix saved to {cm_path}")


def quick_train():
    """Quick training function for testing."""
    print("Starting quick training with mock data...")
    
    # Quick configuration
    config = {
        'model': 'simple_3dcnn',
        'num_classes': 5,
        'epochs': 5,
        'batch_size': 16,
        'learning_rate': 0.001,
        'num_frames': 16,
        'resize': (112, 112),
        'num_train': 200,
        'num_val': 50,
        'save_dir': 'quick_checkpoints',
        'use_wandb': False
    }
    
    # Create model
    model = create_model(config['model'], num_classes=config['num_classes'])
    print(f"Created {config['model']} model")
    
    # Create data loaders
    train_loader, val_loader = create_mock_data_loaders(
        num_train=config['num_train'],
        num_val=config['num_val'],
        num_classes=config['num_classes'],
        batch_size=config['batch_size'],
        num_frames=config['num_frames'],
        resize=config['resize']
    )
    
    # Train
    trainer = VideoTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=config['num_classes'],
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        config=config
    )
    
    trainer.train(epochs=config['epochs'], save_dir=config['save_dir'])
    
    # Evaluate
    results = trainer.evaluate()
    print(f"Final accuracy: {results.get('accuracy', 0):.4f}")


if __name__ == '__main__':
    # Check if this is a quick test run
    import sys
    if len(sys.argv) == 1:
        print("No arguments provided. Running quick training...")
        quick_train()
    else:
        main()
