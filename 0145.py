# Project 145. Video classification with 3D CNN - MODERN IMPLEMENTATION
# Description:
# Video classification involves identifying actions, events, or objects in a video sequence. 
# Unlike image classification, it requires understanding both spatial and temporal features. 
# This project uses modern 3D Convolutional Neural Networks (3D CNN), including I3D, SlowFast, and X3D.

# 🚀 MODERN FEATURES:
# - Multiple 3D CNN architectures (Simple3DCNN, I3D, SlowFast, X3D)
# - Comprehensive training pipeline with metrics and checkpointing
# - Mock dataset generator for testing
# - Streamlit web interface
# - Configuration management
# - Modern dependencies and best practices

# Quick Start Examples:

# 1. Basic Usage (Updated)
from models import create_model, get_model_info
from data import load_video_tensor, create_mock_data_loaders
from training import VideoTrainer
from configs import create_config_for_model
import torch

# Create a modern 3D CNN model
model = create_model('simple_3dcnn', num_classes=10)
print(f"Model parameters: {get_model_info(model)['total_parameters']:,}")

# Load video (if you have a video file)
# video_tensor = load_video_tensor('sample_video.mp4')  # [1, C, T, H, W]

# For demo purposes, create mock data
train_loader, val_loader = create_mock_data_loaders(
    num_train=100, num_val=50, num_classes=10, batch_size=16
)

# Quick training example
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = VideoTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_classes=10,
    device=device
)

# Train for a few epochs
print("Training model...")
trainer.train(epochs=5, save_dir='demo_checkpoints')

# Get predictions
model.eval()
with torch.no_grad():
    for videos, labels in val_loader:
        videos = videos.to(device)
        outputs = model(videos)
        predictions = torch.argmax(outputs, dim=1)
        print(f"🎥 Sample predictions: {predictions[:5].tolist()}")
        break

# 2. Web Interface
print("\n🌐 To run the web interface:")
print("streamlit run app.py")

# 3. Command Line Training
print("\n🏋️ To train from command line:")
print("python train.py --model simple_3dcnn --epochs 50 --batch_size 32")

# 4. Available Models
print("\n📋 Available models:")
models = ['simple_3dcnn', 'i3d', 'slowfast', 'x3d']
for model_name in models:
    model = create_model(model_name, num_classes=10)
    info = get_model_info(model)
    print(f"- {model_name}: {info['total_parameters']:,} parameters, {info['model_size_mb']:.1f} MB")

print("\n✨ Modern implementation complete!")
print("Check the README.md for detailed usage instructions.")