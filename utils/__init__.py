"""
Utility functions for video classification project.
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict, Any
import os
from pathlib import Path


def visualize_video_frames(video_tensor: torch.Tensor, num_frames: int = 8, 
                          save_path: Optional[str] = None) -> None:
    """
    Visualize frames from a video tensor.
    
    Args:
        video_tensor: Video tensor [C, T, H, W] or [B, C, T, H, W]
        num_frames: Number of frames to display
        save_path: Path to save the visualization
    """
    # Handle batch dimension
    if video_tensor.dim() == 5:
        video_tensor = video_tensor[0]  # Take first sample
    
    # Convert to numpy
    frames = video_tensor.permute(1, 2, 3, 0).numpy()  # [T, H, W, C]
    
    # Denormalize if needed
    if frames.min() < 0:
        frames = (frames + 1) / 2  # Assume normalization to [-1, 1]
    frames = np.clip(frames, 0, 1)
    
    # Select frames to display
    total_frames = frames.shape[0]
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    selected_frames = frames[frame_indices]
    
    # Create subplot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, frame in enumerate(selected_frames):
        if i < len(axes):
            axes[i].imshow(frame)
            axes[i].set_title(f'Frame {frame_indices[i]}')
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(selected_frames), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_video_from_frames(frames: np.ndarray, output_path: str, 
                            fps: int = 30) -> None:
    """
    Create a video file from frame array.
    
    Args:
        frames: Frame array [T, H, W, C]
        output_path: Output video path
        fps: Frames per second
    """
    height, width = frames.shape[1], frames.shape[2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()


def analyze_model_complexity(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Analyze model complexity and performance characteristics.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with complexity metrics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count layers by type
    layer_counts = {}
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if module_type not in layer_counts:
            layer_counts[module_type] = 0
        layer_counts[module_type] += 1
    
    # Estimate FLOPs (rough approximation)
    def count_flops(module, input_shape):
        if isinstance(module, torch.nn.Conv3d):
            # Conv3d FLOPs = output_elements * kernel_elements * input_channels
            output_elements = np.prod(input_shape[2:])  # T*H*W
            kernel_elements = module.kernel_size[0] * module.kernel_size[1] * module.kernel_size[2]
            return output_elements * kernel_elements * module.in_channels
        elif isinstance(module, torch.nn.Linear):
            return module.in_features * module.out_features
        return 0
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        'layer_counts': layer_counts,
        'memory_efficient': total_params < 10_000_000  # Less than 10M params
    }


def plot_training_curves(history: Dict[str, List], save_path: Optional[str] = None):
    """
    Plot training curves from history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0].plot(history['val_loss'], label='Validation Loss', color='red')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Accuracy', color='blue')
    axes[1].plot(history['val_acc'], label='Validation Accuracy', color='red')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix_advanced(y_true: List[int], y_pred: List[int], 
                                  class_names: Optional[List[str]] = None,
                                  save_path: Optional[str] = None):
    """
    Plot advanced confusion matrix with additional metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class names
        save_path: Path to save the plot
    """
    from sklearn.metrics import confusion_matrix, classification_report
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Confusion matrix with counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # Confusion matrix with percentages
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', ax=ax2)
    ax2.set_title('Confusion Matrix (Percentages)')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    if class_names:
        ax1.set_xticklabels(class_names, rotation=45)
        ax1.set_yticklabels(class_names, rotation=0)
        ax2.set_xticklabels(class_names, rotation=45)
        ax2.set_yticklabels(class_names, rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))


def benchmark_model_speed(model: torch.nn.Module, input_shape: Tuple[int, ...], 
                         device: str = 'cpu', num_runs: int = 100) -> Dict[str, float]:
    """
    Benchmark model inference speed.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        device: Device to run on
        num_runs: Number of runs for averaging
        
    Returns:
        Dictionary with timing results
    """
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Benchmark
    import time
    
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(dummy_input)
            end_time = time.time()
            times.append(end_time - start_time)
    
    times = np.array(times)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'fps': 1.0 / np.mean(times),
        'throughput': input_shape[0] / np.mean(times)  # samples per second
    }


def save_model_summary(model: torch.nn.Module, save_path: str):
    """
    Save a comprehensive model summary.
    
    Args:
        model: PyTorch model
        save_path: Path to save the summary
    """
    with open(save_path, 'w') as f:
        f.write("Model Summary\n")
        f.write("=" * 50 + "\n\n")
        
        # Model architecture
        f.write("Architecture:\n")
        f.write(str(model) + "\n\n")
        
        # Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n")
        f.write(f"Model Size: {total_params * 4 / (1024 * 1024):.2f} MB\n\n")
        
        # Layer breakdown
        f.write("Layer Breakdown:\n")
        layer_counts = {}
        for name, module in model.named_modules():
            module_type = type(module).__name__
            if module_type not in layer_counts:
                layer_counts[module_type] = 0
            layer_counts[module_type] += 1
        
        for layer_type, count in layer_counts.items():
            f.write(f"  {layer_type}: {count}\n")


def create_gif_from_video(video_path: str, output_path: str, 
                         max_frames: int = 30, fps: int = 10):
    """
    Create a GIF from video file.
    
    Args:
        video_path: Input video path
        output_path: Output GIF path
        max_frames: Maximum number of frames
        fps: Frames per second for GIF
    """
    import imageio
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // max_frames, 1)
    
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        if len(frames) >= max_frames:
            break
    
    cap.release()
    
    # Save as GIF
    imageio.mimsave(output_path, frames, fps=fps)


def setup_project_directories():
    """Setup project directory structure."""
    directories = [
        'checkpoints',
        'data',
        'logs',
        'results',
        'configs',
        'models',
        'training',
        'utils'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Project directories created successfully!")


if __name__ == '__main__':
    setup_project_directories()
