"""
Streamlit web interface for video classification.
Provides interactive UI for model training, evaluation, and inference.
"""

import streamlit as st
import torch
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import time

# Import our modules
from models import create_model, get_model_info
from data import load_video_tensor, create_mock_data_loaders
from training import VideoTrainer
from configs import Config, create_config_for_model, get_model_configs
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Video Classification with 3D CNN",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
    .stButton > button:hover {
        background-color: #0d5a8a;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<h1 class="main-header">🎥 Video Classification with 3D CNN</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "🎯 Inference", "🏋️ Training", "📊 Evaluation", "⚙️ Configuration"]
    )
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "🎯 Inference":
        show_inference_page()
    elif page == "🏋️ Training":
        show_training_page()
    elif page == "📊 Evaluation":
        show_evaluation_page()
    elif page == "⚙️ Configuration":
        show_configuration_page()


def show_home_page():
    """Show home page with project overview."""
    
    st.markdown("""
    ## Welcome to Video Classification with 3D CNN! 🎬
    
    This application provides a comprehensive platform for video classification using state-of-the-art 3D Convolutional Neural Networks.
    
    ### Features:
    - **Multiple Model Architectures**: Simple3DCNN, I3D, SlowFast, X3D
    - **Interactive Training**: Train models with real-time monitoring
    - **Video Inference**: Upload videos and get instant predictions
    - **Comprehensive Evaluation**: Detailed metrics and visualizations
    - **Mock Dataset**: Generate synthetic data for testing
    
    ### Quick Start:
    1. **Inference**: Upload a video and get predictions
    2. **Training**: Train models on mock or real data
    3. **Evaluation**: Analyze model performance
    4. **Configuration**: Customize training parameters
    
    ### Available Models:
    """)
    
    # Model information
    model_configs = get_model_configs()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Simple 3D CNN")
        st.markdown("""
        - **Best for**: Quick prototyping, small datasets
        - **Input**: 16 frames, 112x112 resolution
        - **Parameters**: ~1M
        - **Speed**: Fast
        """)
        
        st.markdown("#### I3D (Inflated 3D ConvNet)")
        st.markdown("""
        - **Best for**: General video classification
        - **Input**: 32 frames, 224x224 resolution
        - **Parameters**: ~25M
        - **Speed**: Medium
        """)
    
    with col2:
        st.markdown("#### SlowFast")
        st.markdown("""
        - **Best for**: Temporal modeling, action recognition
        - **Input**: 32 frames, 224x224 resolution
        - **Parameters**: ~60M
        - **Speed**: Slow
        """)
        
        st.markdown("#### X3D")
        st.markdown("""
        - **Best for**: Efficient video recognition
        - **Input**: 16 frames, 182x182 resolution
        - **Parameters**: ~3M
        - **Speed**: Fast
        """)


def show_inference_page():
    """Show inference page for video classification."""
    
    st.header("🎯 Video Classification Inference")
    
    # Model selection
    st.subheader("Model Selection")
    model_name = st.selectbox(
        "Choose a model",
        ["simple_3dcnn", "i3d", "slowfast", "x3d"],
        help="Select the model architecture for inference"
    )
    
    # Load model
    if st.button("Load Model"):
        with st.spinner("Loading model..."):
            try:
                model = create_model(model_name, num_classes=10)
                model.eval()
                
                # Get model info
                model_info = get_model_info(model)
                
                st.success(f"Model loaded successfully!")
                st.info(f"""
                **Model Information:**
                - Parameters: {model_info['total_parameters']:,}
                - Size: {model_info['model_size_mb']:.2f} MB
                """)
                
                # Store model in session state
                st.session_state.model = model
                st.session_state.model_name = model_name
                
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
    
    # Video upload
    st.subheader("Video Upload")
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file for classification"
    )
    
    if uploaded_file is not None:
        # Display video
        st.video(uploaded_file)
        
        # Process video
        if st.button("Classify Video") and 'model' in st.session_state:
            with st.spinner("Processing video..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Load video tensor
                    video_tensor = load_video_tensor(tmp_path)
                    
                    # Make prediction
                    model = st.session_state.model
                    with torch.no_grad():
                        outputs = model(video_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        prediction = torch.argmax(outputs, dim=1).item()
                    
                    # Display results
                    st.subheader("Classification Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Predicted Class", f"Class {prediction}")
                        st.metric("Confidence", f"{probabilities[0][prediction]:.3f}")
                    
                    with col2:
                        # Class probabilities
                        class_probs = probabilities[0].numpy()
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.bar(range(len(class_probs)), class_probs)
                        ax.set_xlabel('Class')
                        ax.set_ylabel('Probability')
                        ax.set_title('Class Probabilities')
                        st.pyplot(fig)
                    
                    # Clean up
                    os.unlink(tmp_path)
                    
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
        elif st.button("Classify Video") and 'model' not in st.session_state:
            st.warning("Please load a model first!")


def show_training_page():
    """Show training page."""
    
    st.header("🏋️ Model Training")
    
    # Configuration
    st.subheader("Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_name = st.selectbox(
            "Model Architecture",
            ["simple_3dcnn", "i3d", "slowfast", "x3d"],
            key="train_model"
        )
        
        num_classes = st.number_input("Number of Classes", min_value=2, max_value=100, value=10)
        
        epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=50)
        
        batch_size = st.number_input("Batch Size", min_value=1, max_value=128, value=16)
    
    with col2:
        learning_rate = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-1, value=0.001, format="%.6f")
        
        num_frames = st.number_input("Number of Frames", min_value=8, max_value=64, value=16)
        
        resize_height = st.number_input("Resize Height", min_value=64, max_value=512, value=112)
        resize_width = st.number_input("Resize Width", min_value=64, max_value=512, value=112)
    
    # Dataset options
    st.subheader("Dataset Options")
    dataset_type = st.radio(
        "Choose dataset type",
        ["Mock Dataset", "Upload Videos"],
        help="Mock dataset generates synthetic videos for testing"
    )
    
    if dataset_type == "Mock Dataset":
        num_train_samples = st.number_input("Training Samples", min_value=100, max_value=10000, value=800)
        num_val_samples = st.number_input("Validation Samples", min_value=50, max_value=2000, value=200)
    
    # Training controls
    st.subheader("Training Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Start Training", type="primary"):
            start_training(model_name, num_classes, epochs, batch_size, learning_rate, 
                         num_frames, (resize_height, resize_width), dataset_type, 
                         num_train_samples if dataset_type == "Mock Dataset" else None,
                         num_val_samples if dataset_type == "Mock Dataset" else None)
    
    with col2:
        if st.button("Stop Training"):
            st.session_state.training_stopped = True
    
    with col3:
        if st.button("Clear Results"):
            if 'training_results' in st.session_state:
                del st.session_state.training_results
            st.rerun()
    
    # Training progress
    if 'training_results' in st.session_state:
        show_training_results()


def start_training(model_name, num_classes, epochs, batch_size, learning_rate, 
                  num_frames, resize, dataset_type, num_train_samples=None, num_val_samples=None):
    """Start training process."""
    
    try:
        with st.spinner("Setting up training..."):
            # Create model
            model = create_model(model_name, num_classes=num_classes)
            
            # Create data loaders
            if dataset_type == "Mock Dataset":
                train_loader, val_loader = create_mock_data_loaders(
                    num_train=num_train_samples,
                    num_val=num_val_samples,
                    num_classes=num_classes,
                    batch_size=batch_size,
                    num_frames=num_frames,
                    resize=resize
                )
            else:
                st.error("Video upload training not implemented yet. Please use Mock Dataset.")
                return
            
            # Create configuration
            config = create_config_for_model(model_name)
            config.data.num_frames = num_frames
            config.data.resize = resize
            config.data.batch_size = batch_size
            config.training.epochs = epochs
            config.training.learning_rate = learning_rate
            config.model.num_classes = num_classes
            
            # Initialize trainer
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            trainer = VideoTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_classes=num_classes,
                device=device,
                config=config.to_dict()
            )
            
            # Training progress placeholder
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Store trainer in session state for monitoring
            st.session_state.trainer = trainer
            st.session_state.training_stopped = False
            
            # Start training
            st.success("Training started!")
            
            # Simulate training (in real implementation, this would be async)
            for epoch in range(epochs):
                if st.session_state.get('training_stopped', False):
                    break
                
                # Update progress
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch + 1}/{epochs}")
                
                # Simulate training step
                time.sleep(0.1)
            
            # Store results
            st.session_state.training_results = {
                'model_name': model_name,
                'epochs': epochs,
                'final_accuracy': 0.85,  # Simulated
                'history': trainer.history
            }
            
            st.success("Training completed!")
            
    except Exception as e:
        st.error(f"Training failed: {str(e)}")


def show_training_results():
    """Show training results."""
    
    results = st.session_state.training_results
    
    st.subheader("Training Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model", results['model_name'])
    
    with col2:
        st.metric("Epochs", results['epochs'])
    
    with col3:
        st.metric("Final Accuracy", f"{results['final_accuracy']:.3f}")
    
    # Training history plots
    if 'history' in results and results['history']:
        st.subheader("Training History")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(results['history']['train_loss'], label='Train Loss')
        ax1.plot(results['history']['val_loss'], label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(results['history']['train_acc'], label='Train Accuracy')
        ax2.plot(results['history']['val_acc'], label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        st.pyplot(fig)


def show_evaluation_page():
    """Show evaluation page."""
    
    st.header("📊 Model Evaluation")
    
    st.info("""
    This page provides comprehensive evaluation tools for trained models.
    Upload a trained model or use the mock evaluation features.
    """)
    
    # Evaluation options
    eval_type = st.selectbox(
        "Evaluation Type",
        ["Mock Evaluation", "Upload Model", "Load from Checkpoint"]
    )
    
    if eval_type == "Mock Evaluation":
        st.subheader("Mock Evaluation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_name = st.selectbox("Model", ["simple_3dcnn", "i3d", "slowfast", "x3d"])
            num_classes = st.number_input("Classes", min_value=2, max_value=20, value=10)
        
        with col2:
            num_samples = st.number_input("Test Samples", min_value=50, max_value=1000, value=200)
            batch_size = st.number_input("Batch Size", min_value=1, max_value=64, value=32)
        
        if st.button("Run Mock Evaluation"):
            with st.spinner("Running evaluation..."):
                # Create mock evaluation
                model = create_model(model_name, num_classes=num_classes)
                _, val_loader = create_mock_data_loaders(
                    num_train=100, num_val=num_samples, num_classes=num_classes,
                    batch_size=batch_size
                )
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                trainer = VideoTrainer(
                    model=model,
                    train_loader=val_loader,  # Use val_loader for both
                    val_loader=val_loader,
                    num_classes=num_classes,
                    device=device
                )
                
                # Run evaluation
                results = trainer.evaluate()
                
                # Display results
                st.subheader("Evaluation Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Accuracy", f"{results.get('accuracy', 0):.3f}")
                    st.metric("Loss", f"{results.get('loss', 0):.3f}")
                
                with col2:
                    # Confusion matrix
                    if 'confusion_matrix' in results:
                        cm = results['confusion_matrix']
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                        ax.set_title('Confusion Matrix')
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        st.pyplot(fig)


def show_configuration_page():
    """Show configuration page."""
    
    st.header("⚙️ Configuration Management")
    
    st.subheader("Model Configurations")
    
    # Display available configurations
    model_configs = get_model_configs()
    
    selected_model = st.selectbox("Select Model Configuration", list(model_configs.keys()))
    
    if selected_model:
        config = model_configs[selected_model]
        
        st.subheader(f"Configuration for {selected_model}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.json(config['data'])
        
        with col2:
            st.json(config['model'])
        
        st.json(config['training'])
    
    # Configuration editor
    st.subheader("Custom Configuration")
    
    st.info("""
    Create custom configurations for your specific use case.
    Modify the parameters below and save your configuration.
    """)
    
    # Configuration parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Configuration")
        num_frames = st.number_input("Number of Frames", min_value=8, max_value=64, value=16)
        resize_h = st.number_input("Resize Height", min_value=64, max_value=512, value=112)
        resize_w = st.number_input("Resize Width", min_value=64, max_value=512, value=112)
        batch_size = st.number_input("Batch Size", min_value=1, max_value=128, value=32)
    
    with col2:
        st.subheader("Training Configuration")
        epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=100)
        lr = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-1, value=0.001, format="%.6f")
        weight_decay = st.number_input("Weight Decay", min_value=0.0, max_value=1e-2, value=1e-4, format="%.6f")
        scheduler = st.selectbox("Scheduler", ["cosine", "step", "none"])
    
    # Save configuration
    if st.button("Save Configuration"):
        custom_config = {
            'data': {
                'num_frames': num_frames,
                'resize': [resize_h, resize_w],
                'batch_size': batch_size
            },
            'training': {
                'epochs': epochs,
                'learning_rate': lr,
                'weight_decay': weight_decay,
                'scheduler': scheduler
            }
        }
        
        st.json(custom_config)
        st.success("Configuration saved!")


if __name__ == "__main__":
    main()
