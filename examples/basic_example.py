"""
Basic example demonstrating video classification with 3D CNN.
"""

import torch
import numpy as np
from models import create_model, get_model_info
from data import create_mock_data_loaders, load_video_tensor
from training import VideoTrainer
from configs import create_config_for_model
from utils import visualize_video_frames, plot_training_curves
import matplotlib.pyplot as plt


def main():
    """Basic example workflow."""
    print("🎥 Video Classification with 3D CNN - Basic Example")
    print("=" * 60)
    
    # 1. Create model
    print("\n1. Creating model...")
    model_name = 'simple_3dcnn'
    num_classes = 10
    
    model = create_model(model_name, num_classes=num_classes)
    model_info = get_model_info(model)
    
    print(f"✅ Created {model_name} model")
    print(f"   Parameters: {model_info['total_parameters']:,}")
    print(f"   Size: {model_info['model_size_mb']:.2f} MB")
    
    # 2. Create mock dataset
    print("\n2. Creating mock dataset...")
    train_loader, val_loader = create_mock_data_loaders(
        num_train=200,
        num_val=50,
        num_classes=num_classes,
        batch_size=16,
        num_frames=16,
        resize=(112, 112)
    )
    
    print(f"✅ Created dataset: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val samples")
    
    # 3. Visualize sample data
    print("\n3. Visualizing sample data...")
    for videos, labels in train_loader:
        print(f"   Video shape: {videos.shape}")
        print(f"   Labels: {labels[:5].tolist()}")
        
        # Visualize first video
        visualize_video_frames(videos[0], num_frames=8)
        break
    
    # 4. Train model
    print("\n4. Training model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    trainer = VideoTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        device=device
    )
    
    # Train for a few epochs
    history = trainer.train(epochs=5, save_dir='example_checkpoints')
    
    # 5. Evaluate model
    print("\n5. Evaluating model...")
    results = trainer.evaluate()
    
    print(f"✅ Final accuracy: {results.get('accuracy', 0):.3f}")
    print(f"   Final loss: {results.get('loss', 0):.3f}")
    
    # 6. Make predictions
    print("\n6. Making predictions...")
    model.eval()
    with torch.no_grad():
        for videos, labels in val_loader:
            videos = videos.to(device)
            outputs = model(videos)
            predictions = torch.argmax(outputs, dim=1)
            
            print(f"   Sample predictions: {predictions[:5].tolist()}")
            print(f"   True labels: {labels[:5].tolist()}")
            break
    
    # 7. Plot results
    print("\n7. Plotting training results...")
    plot_training_curves(history)
    
    print("\n✅ Basic example completed successfully!")
    print("\nNext steps:")
    print("- Try different models: python -c \"from models import create_model; print([create_model(m, 10) for m in ['i3d', 'slowfast', 'x3d']])\"")
    print("- Run web interface: streamlit run app.py")
    print("- Train longer: python train.py --epochs 50")


def compare_models():
    """Compare different model architectures."""
    print("\n🔍 Model Comparison")
    print("=" * 30)
    
    models = ['simple_3dcnn', 'i3d', 'slowfast', 'x3d']
    
    for model_name in models:
        try:
            model = create_model(model_name, num_classes=10)
            info = get_model_info(model)
            
            print(f"{model_name:12} | {info['total_parameters']:>8,} params | {info['model_size_mb']:>6.1f} MB")
        except Exception as e:
            print(f"{model_name:12} | Error: {str(e)}")


def quick_inference_example():
    """Quick inference example."""
    print("\n🎯 Quick Inference Example")
    print("=" * 30)
    
    # Create model
    model = create_model('simple_3dcnn', num_classes=5)
    model.eval()
    
    # Create mock video tensor
    video_tensor = torch.randn(1, 3, 16, 112, 112)  # [B, C, T, H, W]
    
    # Make prediction
    with torch.no_grad():
        outputs = model(video_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(outputs, dim=1).item()
    
    print(f"Predicted class: {prediction}")
    print(f"Confidence: {probabilities[0][prediction]:.3f}")
    print(f"All probabilities: {probabilities[0].tolist()}")


if __name__ == '__main__':
    main()
    compare_models()
    quick_inference_example()
