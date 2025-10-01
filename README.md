# Video Classification with 3D CNN

A modern, comprehensive implementation of 3D Convolutional Neural Networks for video classification, featuring multiple architectures, robust training pipeline, and interactive web interface.

## Features

- **Multiple 3D CNN Architectures**: Simple3DCNN, I3D, SlowFast, X3D
- **Comprehensive Training Pipeline**: Training, validation, metrics, and checkpointing
- **Mock Dataset Generator**: Create synthetic video data for testing
- **Interactive Web Interface**: Streamlit-based UI for training and inference
- **Modern Dependencies**: Latest PyTorch, OpenCV, and ML libraries
- **Configuration Management**: Flexible configuration system
- **Evaluation Tools**: Comprehensive metrics and visualizations

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Video-Classification-with-3D-CNN.git
cd Video-Classification-with-3D-CNN

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Web Interface

```bash
streamlit run app.py
```

### 3. Quick Training

```bash
# Train with default settings
python train.py

# Train specific model
python train.py --model simple_3dcnn --epochs 50 --batch_size 32
```

### 4. Basic Usage

```python
from models import create_model
from data import create_mock_data_loaders
from training import VideoTrainer

# Create model
model = create_model('simple_3dcnn', num_classes=10)

# Create data loaders
train_loader, val_loader = create_mock_data_loaders(
    num_train=800, num_val=200, num_classes=10
)

# Train model
trainer = VideoTrainer(model, train_loader, val_loader, num_classes=10)
trainer.train(epochs=50)
```

## Project Structure

```
├── models/              # Model implementations
│   └── __init__.py     # Simple3DCNN, I3D, SlowFast, X3D
├── data/               # Data loading and preprocessing
│   └── __init__.py     # VideoDataset, MockVideoDataset
├── training/           # Training pipeline
│   └── __init__.py     # VideoTrainer, Metrics
├── configs/            # Configuration management
│   └── __init__.py     # Config classes and utilities
├── utils/              # Utility functions
│   └── __init__.py     # Visualization, analysis tools
├── app.py              # Streamlit web interface
├── train.py            # Training script
├── 0145.py            # Updated original implementation
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## Available Models

| Model | Parameters | Size | Best For | Speed |
|-------|------------|------|----------|-------|
| **Simple3DCNN** | ~1M | 4MB | Quick prototyping | Fast |
| **I3D** | ~25M | 100MB | General classification | Medium |
| **SlowFast** | ~60M | 240MB | Temporal modeling | Slow |
| **X3D** | ~3M | 12MB | Efficient recognition | Fast |

## Web Interface

The Streamlit web interface provides:

- **Home**: Project overview and model information
- **Inference**: Upload videos and get predictions
- **Training**: Interactive model training
- **Evaluation**: Comprehensive model evaluation
- **Configuration**: Manage training parameters

## Training Options

### Command Line Training

```bash
# Basic training
python train.py --model simple_3dcnn --epochs 100

# Advanced training
python train.py \
    --model i3d \
    --epochs 200 \
    --batch_size 16 \
    --lr 0.0001 \
    --num_frames 32 \
    --resize 224 224
```

### Configuration Files

```bash
# Use configuration file
python train.py --config configs/i3d.yaml

# Create custom configuration
python -c "from configs import setup_configs; setup_configs()"
```

## 🔧 Configuration

The project uses a flexible configuration system:

```python
from configs import create_config_for_model

# Create model-specific config
config = create_config_for_model('i3d', num_classes=20)

# Modify parameters
config.training.learning_rate = 0.0001
config.data.num_frames = 32

# Save configuration
config.save('my_config.yaml')
```

## Evaluation

### Metrics

- **Accuracy**: Overall classification accuracy
- **Per-class Accuracy**: Individual class performance
- **Confusion Matrix**: Detailed classification analysis
- **Training Curves**: Loss and accuracy over time

### Visualization

```python
from utils import plot_training_curves, plot_confusion_matrix_advanced

# Plot training history
plot_training_curves(trainer.history)

# Advanced confusion matrix
plot_confusion_matrix_advanced(y_true, y_pred, class_names)
```

## Mock Dataset

Generate synthetic video data for testing:

```python
from data import MockVideoDataset

# Create mock dataset
dataset = MockVideoDataset(
    num_samples=1000,
    num_classes=10,
    num_frames=16,
    resize=(112, 112)
)

# Use in training
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## Advanced Usage

### Custom Model

```python
from models import create_model

# Create custom model
model = create_model('simple_3dcnn', num_classes=20, dropout_rate=0.3)
```

### Custom Training

```python
from training import VideoTrainer

# Custom training configuration
config = {
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'scheduler': 'cosine',
    'use_wandb': True
}

trainer = VideoTrainer(model, train_loader, val_loader, num_classes=10, config=config)
trainer.train(epochs=100)
```

### Model Analysis

```python
from utils import analyze_model_complexity, benchmark_model_speed

# Analyze model
complexity = analyze_model_complexity(model)
print(f"Parameters: {complexity['total_parameters']:,}")

# Benchmark speed
speed = benchmark_model_speed(model, input_shape=(1, 3, 16, 112, 112))
print(f"FPS: {speed['fps']:.2f}")
```

## Examples

### 1. Quick Demo

```python
# Run the updated original file
python 0145.py
```

### 2. Web Interface Demo

```bash
streamlit run app.py
# Navigate to http://localhost:8501
```

### 3. Full Training Pipeline

```bash
# Train I3D model
python train.py --model i3d --epochs 50 --batch_size 16

# Evaluate results
python -c "
from training import VideoTrainer
from models import create_model
from data import create_mock_data_loaders

model = create_model('i3d', num_classes=10)
train_loader, val_loader = create_mock_data_loaders(100, 50, 10, 16)
trainer = VideoTrainer(model, train_loader, val_loader, 10)
results = trainer.evaluate()
print(f'Accuracy: {results[\"accuracy\"]:.3f}')
"
```

## 🛠️ Development

### Adding New Models

1. Implement model in `models/__init__.py`
2. Add to `create_model()` function
3. Update configuration in `configs/__init__.py`
4. Test with mock data

### Adding New Features

1. Implement in appropriate module
2. Update web interface if needed
3. Add configuration options
4. Update documentation

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the examples

## Acknowledgments

- PyTorch team for the excellent framework
- OpenCV for video processing capabilities
- Streamlit for the web interface
- The computer vision community for 3D CNN research


# Video-Classification-with-3D-CNN
