"""
Setup script for video classification project.
Creates necessary directories and initializes the project.
"""

import os
import sys
from pathlib import Path


def create_directories():
    """Create necessary project directories."""
    directories = [
        'checkpoints',
        'data',
        'logs',
        'results',
        'configs',
        'models',
        'training',
        'utils',
        'examples'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Create __init__.py files for Python packages
    init_files = [
        'models/__init__.py',
        'data/__init__.py',
        'training/__init__.py',
        'configs/__init__.py',
        'utils/__init__.py',
        'examples/__init__.py'
    ]
    
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('"""Package initialization."""\n')
            print(f"✅ Created: {init_file}")


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'torch',
        'torchvision',
        'opencv-python',
        'numpy',
        'matplotlib',
        'seaborn',
        'streamlit',
        'albumentations',
        'scikit-learn',
        'pandas',
        'tqdm',
        'pillow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required packages are installed!")
        return True


def create_sample_configs():
    """Create sample configuration files."""
    from configs import setup_configs
    
    try:
        setup_configs()
        print("✅ Configuration files created")
    except Exception as e:
        print(f"⚠️  Could not create configs: {e}")


def test_basic_functionality():
    """Test basic functionality."""
    try:
        from models import create_model, get_model_info
        from data import create_mock_data_loaders
        
        # Test model creation
        model = create_model('simple_3dcnn', num_classes=5)
        info = get_model_info(model)
        print(f"✅ Model creation test passed: {info['total_parameters']:,} parameters")
        
        # Test data loading
        train_loader, val_loader = create_mock_data_loaders(
            num_train=10, num_val=5, num_classes=5, batch_size=2
        )
        print(f"✅ Data loading test passed: {len(train_loader.dataset)} train samples")
        
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Video Classification with 3D CNN project...")
    print("=" * 60)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    deps_ok = check_dependencies()
    
    # Create sample configs
    print("\n⚙️  Creating configuration files...")
    create_sample_configs()
    
    # Test basic functionality
    print("\n🧪 Testing basic functionality...")
    if deps_ok:
        test_ok = test_basic_functionality()
    else:
        test_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Setup Summary:")
    print("=" * 60)
    
    if deps_ok and test_ok:
        print("✅ Setup completed successfully!")
        print("\n🎯 Next steps:")
        print("1. Run web interface: streamlit run app.py")
        print("2. Try basic example: python examples/basic_example.py")
        print("3. Train a model: python train.py --epochs 10")
        print("4. Read documentation: README.md")
    else:
        print("⚠️  Setup completed with issues.")
        if not deps_ok:
            print("- Install missing dependencies: pip install -r requirements.txt")
        if not test_ok:
            print("- Check error messages above")
    
    print("\n🎉 Happy video classifying!")


if __name__ == '__main__':
    main()
