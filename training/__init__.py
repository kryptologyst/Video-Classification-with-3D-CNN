"""
Training pipeline for video classification models.
Includes trainer class, metrics, checkpointing, and logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class Metrics:
    """
    Metrics calculator for video classification.
    """
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.losses = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, loss: float):
        """
        Update metrics with batch predictions and targets.
        
        Args:
            predictions: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            loss: Loss value
        """
        pred_labels = torch.argmax(predictions, dim=1).cpu().numpy()
        self.predictions.extend(pred_labels)
        self.targets.extend(targets.cpu().numpy())
        self.losses.append(loss)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.
        
        Returns:
            Dictionary of metrics
        """
        if not self.predictions:
            return {}
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Accuracy
        accuracy = np.mean(predictions == targets)
        
        # Average loss
        avg_loss = np.mean(self.losses)
        
        # Per-class accuracy
        class_accuracies = []
        for i in range(self.num_classes):
            mask = targets == i
            if np.sum(mask) > 0:
                class_acc = np.mean(predictions[mask] == i)
                class_accuracies.append(class_acc)
            else:
                class_accuracies.append(0.0)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'class_accuracies': class_accuracies,
            'predictions': predictions,
            'targets': targets
        }


class VideoTrainer:
    """
    Trainer class for video classification models.
    """
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 num_classes: int, device: str = 'cuda', config: Optional[Dict] = None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.device = device
        self.config = config or {}
        
        # Initialize metrics
        self.train_metrics = Metrics(num_classes)
        self.val_metrics = Metrics(num_classes)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Setup optimizer and scheduler
        self.setup_optimizer()
        self.setup_scheduler()
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup logging
        self.setup_logging()
    
    def setup_optimizer(self):
        """Setup optimizer."""
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    
    def setup_scheduler(self):
        """Setup learning rate scheduler."""
        scheduler_type = self.config.get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 100)
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=self.config.get('gamma', 0.1)
            )
        else:
            self.scheduler = None
    
    def setup_logging(self):
        """Setup logging (wandb, tensorboard, etc.)."""
        self.use_wandb = self.config.get('use_wandb', False)
        
        if self.use_wandb:
            wandb.init(
                project=self.config.get('project_name', 'video-classification'),
                config=self.config
            )
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.train_metrics.reset()
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (videos, labels) in enumerate(pbar):
            videos = videos.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(videos)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            self.train_metrics.update(outputs, labels, loss.item())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute metrics
        metrics = self.train_metrics.compute()
        return metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        self.val_metrics.reset()
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for videos, labels in pbar:
                videos = videos.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(videos)
                loss = self.criterion(outputs, labels)
                
                # Update metrics
                self.val_metrics.update(outputs, labels, loss.item())
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute metrics
        metrics = self.val_metrics.compute()
        return metrics
    
    def train(self, epochs: int, save_dir: str = 'checkpoints', 
              save_best: bool = True, patience: int = 10) -> Dict[str, List]:
        """
        Train the model.
        
        Args:
            epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            save_best: Whether to save best model
            patience: Early stopping patience
            
        Returns:
            Training history
        """
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_acc = 0.0
        patience_counter = 0
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_metrics = self.train_epoch()
            train_acc = train_metrics['accuracy']
            train_loss = train_metrics['loss']
            
            # Validate
            val_metrics = self.validate_epoch()
            val_acc = val_metrics['accuracy']
            val_loss = val_metrics['loss']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'learning_rate': current_lr
                })
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'history': self.history
            }
            
            # Save best model
            if save_best and val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
                print(f"New best model saved! Val Acc: {val_acc:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save latest checkpoint
            torch.save(checkpoint, os.path.join(save_dir, 'latest_model.pth'))
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement")
                break
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        
        return self.history
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.history['train_acc'], label='Train Accuracy')
        ax2.plot(self.history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None):
        """
        Plot confusion matrix.
        
        Args:
            save_path: Path to save the plot
        """
        val_metrics = self.val_metrics.compute()
        if not val_metrics:
            print("No validation metrics available")
            return
        
        cm = confusion_matrix(val_metrics['targets'], val_metrics['predictions'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model on validation set.
        
        Returns:
            Evaluation results
        """
        val_metrics = self.validate_epoch()
        
        if not val_metrics:
            return {}
        
        # Classification report
        report = classification_report(
            val_metrics['targets'],
            val_metrics['predictions'],
            output_dict=True
        )
        
        return {
            'accuracy': val_metrics['accuracy'],
            'loss': val_metrics['loss'],
            'classification_report': report,
            'confusion_matrix': confusion_matrix(val_metrics['targets'], val_metrics['predictions'])
        }
    
    def save_model(self, path: str):
        """
        Save the model.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'history': self.history
        }, path)
    
    def load_model(self, path: str):
        """
        Load the model.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        print(f"Model loaded from {path}")


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                config: Dict[str, Any]) -> VideoTrainer:
    """
    Train a video classification model.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        
    Returns:
        Trained VideoTrainer instance
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = config.get('num_classes', 10)
    
    trainer = VideoTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        device=device,
        config=config
    )
    
    epochs = config.get('epochs', 100)
    save_dir = config.get('save_dir', 'checkpoints')
    
    trainer.train(epochs=epochs, save_dir=save_dir)
    
    return trainer
