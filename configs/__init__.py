"""
Configuration management for video classification project.
Uses Hydra for flexible configuration handling.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import yaml
import os
from pathlib import Path


@dataclass
class DataConfig:
    """Data configuration."""
    num_frames: int = 16
    resize: Tuple[int, int] = (112, 112)
    batch_size: int = 32
    num_workers: int = 4
    augment: bool = True
    temporal_stride: int = 1
    num_classes: int = 10


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = 'simple_3dcnn'
    num_classes: int = 10
    dropout_rate: float = 0.5
    # Model-specific parameters
    alpha: int = 8  # For SlowFast
    beta: int = 8   # For SlowFast
    model_size: str = 'S'  # For X3D


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    scheduler: str = 'cosine'  # 'cosine', 'step', None
    step_size: int = 30
    gamma: float = 0.1
    patience: int = 10
    save_dir: str = 'checkpoints'


@dataclass
class LoggingConfig:
    """Logging configuration."""
    use_wandb: bool = False
    project_name: str = 'video-classification'
    log_interval: int = 10
    save_plots: bool = True


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    logging: LoggingConfig = LoggingConfig()
    
    # Global settings
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'
    seed: int = 42
    debug: bool = False
    
    def __post_init__(self):
        """Post-initialization setup."""
        if self.device == 'auto':
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'logging': self.logging.__dict__,
            'device': self.device,
            'seed': self.seed,
            'debug': self.debug
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        
        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            logging=logging_config,
            device=config_dict.get('device', 'auto'),
            seed=config_dict.get('seed', 42),
            debug=config_dict.get('debug', False)
        )
    
    def save(self, path: str):
        """Save config to YAML file."""
        config_dict = self.to_dict()
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load config from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)


def create_default_config() -> Config:
    """Create default configuration."""
    return Config()


def create_config_for_model(model_name: str, **kwargs) -> Config:
    """
    Create configuration optimized for specific model.
    
    Args:
        model_name: Name of the model
        **kwargs: Additional configuration parameters
        
    Returns:
        Optimized configuration
    """
    config = Config()
    
    # Model-specific optimizations
    if model_name == 'simple_3dcnn':
        config.data.num_frames = 16
        config.data.resize = (112, 112)
        config.training.learning_rate = 0.001
        config.training.batch_size = 32
        
    elif model_name == 'i3d':
        config.data.num_frames = 32
        config.data.resize = (224, 224)
        config.training.learning_rate = 0.0001
        config.training.batch_size = 16
        config.training.epochs = 200
        
    elif model_name == 'slowfast':
        config.data.num_frames = 32
        config.data.resize = (224, 224)
        config.training.learning_rate = 0.0001
        config.training.batch_size = 8
        config.training.epochs = 200
        config.model.alpha = 8
        config.model.beta = 8
        
    elif model_name == 'x3d':
        config.data.num_frames = 16
        config.data.resize = (182, 182)
        config.training.learning_rate = 0.0001
        config.training.batch_size = 32
        config.training.epochs = 150
        config.model.model_size = 'S'
    
    # Apply additional kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.data, key):
            setattr(config.data, key, value)
        elif hasattr(config.model, key):
            setattr(config.model, key, value)
        elif hasattr(config.training, key):
            setattr(config.training, key, value)
        elif hasattr(config.logging, key):
            setattr(config.logging, key, value)
    
    return config


def get_model_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get predefined configurations for different models.
    
    Returns:
        Dictionary of model configurations
    """
    return {
        'simple_3dcnn': {
            'data': {
                'num_frames': 16,
                'resize': [112, 112],
                'batch_size': 32
            },
            'model': {
                'name': 'simple_3dcnn',
                'dropout_rate': 0.5
            },
            'training': {
                'learning_rate': 0.001,
                'epochs': 100
            }
        },
        'i3d': {
            'data': {
                'num_frames': 32,
                'resize': [224, 224],
                'batch_size': 16
            },
            'model': {
                'name': 'i3d',
                'dropout_rate': 0.5
            },
            'training': {
                'learning_rate': 0.0001,
                'epochs': 200
            }
        },
        'slowfast': {
            'data': {
                'num_frames': 32,
                'resize': [224, 224],
                'batch_size': 8
            },
            'model': {
                'name': 'slowfast',
                'alpha': 8,
                'beta': 8
            },
            'training': {
                'learning_rate': 0.0001,
                'epochs': 200
            }
        },
        'x3d': {
            'data': {
                'num_frames': 16,
                'resize': [182, 182],
                'batch_size': 32
            },
            'model': {
                'name': 'x3d',
                'model_size': 'S'
            },
            'training': {
                'learning_rate': 0.0001,
                'epochs': 150
            }
        }
    }


def save_config_template(path: str = 'configs/template.yaml'):
    """
    Save configuration template.
    
    Args:
        path: Path to save template
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    template_config = create_default_config()
    template_config.save(path)
    
    print(f"Configuration template saved to {path}")


def load_config(path: str) -> Config:
    """
    Load configuration from file.
    
    Args:
        path: Path to configuration file
        
    Returns:
        Loaded configuration
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    return Config.load(path)


# Create configs directory and save templates
def setup_configs():
    """Setup configuration files."""
    configs_dir = Path('configs')
    configs_dir.mkdir(exist_ok=True)
    
    # Save default config
    default_config = create_default_config()
    default_config.save('configs/default.yaml')
    
    # Save model-specific configs
    model_configs = get_model_configs()
    for model_name, config_dict in model_configs.items():
        config = Config.from_dict(config_dict)
        config.save(f'configs/{model_name}.yaml')
    
    # Save template
    save_config_template('configs/template.yaml')
    
    print("Configuration files created in 'configs/' directory")


if __name__ == '__main__':
    setup_configs()
