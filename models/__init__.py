"""
Modern 3D CNN architectures for video classification.
Includes Simple3DCNN, I3D, SlowFast, and X3D implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import math


class Simple3DCNN(nn.Module):
    """
    Enhanced Simple 3D CNN with better architecture and dropout.
    """
    def __init__(self, num_classes: int = 10, input_channels: int = 3, 
                 dropout_rate: float = 0.5):
        super(Simple3DCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(2)
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(2)
        
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(2)
        
        # Global average pooling instead of fixed size
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Classifier
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x: [B, C, T, H, W]
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Global pooling
        x = self.global_pool(x)  # [B, 128, 1, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 128]
        
        x = self.dropout(x)
        return self.fc(x)


class I3D(nn.Module):
    """
    Inflated 3D ConvNet (I3D) - inflates 2D filters to 3D.
    """
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        super(I3D, self).__init__()
        
        # Inflated Inception modules
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(3, stride=2, padding=1)
        
        self.conv2 = nn.Conv3d(64, 192, kernel_size=1)
        self.bn2 = nn.BatchNorm3d(192)
        self.conv3 = nn.Conv3d(192, 192, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(192)
        self.pool2 = nn.MaxPool3d(3, stride=2, padding=1)
        
        # Inception modules
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.pool3 = nn.MaxPool3d(3, stride=2, padding=1)
        
        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.pool4 = nn.MaxPool3d(3, stride=2, padding=1)
        
        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn3(self.conv3(F.relu(self.bn2(self.conv2(x)))))))
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.pool4(x)
        
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


class InceptionModule(nn.Module):
    """Inception module for I3D."""
    def __init__(self, in_channels, out1x1, out3x3_reduce, out3x3, 
                 out3x3_double_reduce, out3x3_double, out_pool):
        super(InceptionModule, self).__init__()
        
        self.branch1 = nn.Conv3d(in_channels, out1x1, kernel_size=1)
        
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, out3x3_reduce, kernel_size=1),
            nn.Conv3d(out3x3_reduce, out3x3, kernel_size=3, padding=1)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, out3x3_double_reduce, kernel_size=1),
            nn.Conv3d(out3x3_double_reduce, out3x3_double, kernel_size=3, padding=1),
            nn.Conv3d(out3x3_double, out3x3_double, kernel_size=3, padding=1)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels, out_pool, kernel_size=1)
        )
        
    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)


class SlowFast(nn.Module):
    """
    SlowFast network with slow and fast pathways.
    """
    def __init__(self, num_classes: int = 10, alpha: int = 8, beta: int = 8):
        super(SlowFast, self).__init__()
        self.alpha = alpha  # temporal stride for slow pathway
        self.beta = beta    # temporal stride for fast pathway
        
        # Slow pathway (low temporal resolution)
        self.slow_conv1 = nn.Conv3d(3, 64, kernel_size=1, stride=1, padding=0)
        self.slow_conv2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.slow_pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        # Fast pathway (high temporal resolution)
        self.fast_conv1 = nn.Conv3d(3, 8, kernel_size=5, stride=1, padding=2)
        self.fast_conv2 = nn.Conv3d(8, 8, kernel_size=3, stride=1, padding=1)
        self.fast_pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        # Lateral connections
        self.lateral_conv = nn.Conv3d(8, 64, kernel_size=5, stride=1, padding=2)
        
        # Fusion layers
        self.fusion_conv = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1)
        
        # Classifier
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        
        # Slow pathway - subsample temporally
        slow_x = x[:, :, ::self.alpha, :, :]  # [B, C, T/alpha, H, W]
        slow_x = F.relu(self.slow_conv1(slow_x))
        slow_x = F.relu(self.slow_conv2(slow_x))
        slow_x = self.slow_pool(slow_x)
        
        # Fast pathway - full temporal resolution
        fast_x = F.relu(self.fast_conv1(x))
        fast_x = F.relu(self.fast_conv2(fast_x))
        fast_x = self.fast_pool(fast_x)
        
        # Lateral connection from fast to slow
        lateral = self.lateral_conv(fast_x)
        
        # Upsample lateral to match slow pathway temporal dimension
        if lateral.shape[2] != slow_x.shape[2]:
            lateral = F.interpolate(lateral, size=(slow_x.shape[2], slow_x.shape[3], slow_x.shape[4]), 
                                  mode='trilinear', align_corners=False)
        
        # Concatenate slow and lateral
        fused = torch.cat([slow_x, lateral], dim=1)
        fused = F.relu(self.fusion_conv(fused))
        
        # Global pooling and classification
        fused = self.global_pool(fused)
        fused = fused.view(fused.size(0), -1)
        return self.fc(fused)


class X3D(nn.Module):
    """
    X3D: Expanding Architectures for Efficient Video Recognition.
    """
    def __init__(self, num_classes: int = 10, model_size: str = 'S'):
        super(X3D, self).__init__()
        
        # Model configurations
        configs = {
            'XS': {'width': 24, 'depth': 4, 'temporal_stride': 1},
            'S': {'width': 48, 'depth': 13, 'temporal_stride': 1},
            'M': {'width': 96, 'depth': 16, 'temporal_stride': 1},
            'L': {'width': 192, 'depth': 16, 'temporal_stride': 1}
        }
        
        config = configs[model_size]
        width = config['width']
        depth = config['depth']
        temporal_stride = config['temporal_stride']
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(3, width, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm3d(width),
            nn.ReLU(inplace=True),
            nn.Conv3d(width, width, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(width),
            nn.ReLU(inplace=True)
        )
        
        # ResNet-like blocks
        self.blocks = nn.ModuleList()
        in_channels = width
        
        for i in range(depth):
            if i == 0:
                stride = (temporal_stride, 2, 2)
            else:
                stride = (1, 2, 2) if i in [depth//4, depth//2, 3*depth//4] else (1, 1, 1)
            
            out_channels = width * (2 ** (i // (depth // 4)))
            
            block = X3DBlock(in_channels, out_channels, stride=stride)
            self.blocks.append(block)
            in_channels = out_channels
        
        # Head
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(in_channels, num_classes)
        
    def forward(self, x):
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class X3DBlock(nn.Module):
    """X3D ResNet block."""
    def __init__(self, in_channels, out_channels, stride=(1, 1, 1)):
        super(X3DBlock, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut connection
        if stride != (1, 1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        out += residual
        return self.relu(out)


def create_model(model_name: str, num_classes: int = 10, **kwargs) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_name: Name of the model ('simple_3dcnn', 'i3d', 'slowfast', 'x3d')
        num_classes: Number of output classes
        **kwargs: Additional model-specific arguments
    
    Returns:
        PyTorch model instance
    """
    models = {
        'simple_3dcnn': Simple3DCNN,
        'i3d': I3D,
        'slowfast': SlowFast,
        'x3d': X3D
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](num_classes=num_classes, **kwargs)


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get model information including parameter count.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
    }
