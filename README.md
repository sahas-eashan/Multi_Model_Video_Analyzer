# Multi-Model Video Analyzer

A sophisticated multimodal sentiment and emotion analysis system that processes text, video, and audio inputs to predict emotional states and sentiment polarities. This project leverages deep learning models to extract and fuse features from multiple modalities for comprehensive video content analysis.

## Overview

The Multi-Model Video Analyzer combines three distinct neural network encoders to analyze:
- **Text**: Using BERT-based language understanding
- **Video**: Using 3D CNN (R3D-18) for temporal visual features
- **Audio**: Using 1D CNN for acoustic feature extraction

The system performs dual-task learning to predict both:
- **Emotion Classification**: 7 classes (anger, disgust, fear, joy, neutral, sadness, surprise)
- **Sentiment Analysis**: 3 classes (negative, neutral, positive)

## Architecture

### Core Components

#### 1. Text Encoder (`TextEncoder`)
- **Base Model**: BERT-base-uncased (frozen parameters)
- **Output Dimension**: 128-dimensional embeddings
- **Features**: Utilizes CLS token representation for sentence-level understanding

#### 2. Video Encoder (`VideoEncoder`)
- **Base Model**: R3D-18 (3D ResNet) pretrained on video datasets
- **Output Dimension**: 128-dimensional embeddings
- **Features**: Processes temporal sequences with 3D convolutions

#### 3. Audio Encoder (`AudioEncoder`)
- **Architecture**: Custom 1D CNN with batch normalization
- **Output Dimension**: 128-dimensional embeddings
- **Features**: Extracts hierarchical acoustic features

#### 4. Fusion Network
- **Input**: Concatenated features (384-dimensional)
- **Architecture**: Multi-layer perceptron with batch normalization and dropout
- **Output**: Dual classification heads for emotion and sentiment

## Technical Specifications

### Dependencies
```python
torch>=1.9.0
transformers>=4.0.0
torchvision>=0.10.0
scikit-learn>=0.24.0
tensorboard>=2.0.0
```

### Key Features
- **Mixed Precision Training**: Supports automatic mixed precision for faster training
- **Gradient Accumulation**: Configurable batch size scaling
- **TensorBoard Logging**: Comprehensive metrics tracking
- **Multi-task Learning**: Joint optimization of emotion and sentiment tasks
- **Frozen Pretrained Components**: Efficient transfer learning approach

### Model Configuration
- **Text Features**: 768 â†’ 128 dimensions
- **Video Features**: R3D-18 backbone â†’ 128 dimensions  
- **Audio Features**: 64-channel input â†’ 128 dimensions
- **Fusion Layer**: 384 â†’ 256 â†’ dual outputs
- **Emotion Classes**: 7 (multi-class classification)
- **Sentiment Classes**: 3 (multi-class classification)

## ðŸ“Š AWS Infrastructure

This project leverages AWS services for scalable machine learning operations:

### Amazon EC2
- **Training Infrastructure**: GPU-enabled instances (p3.xlarge/p3.2xlarge recommended)
- **Development Environment**: Configured with CUDA, PyTorch, and required dependencies
- **Auto-scaling**: Dynamic instance management for training workloads

### Amazon S3
- **Dataset Storage**: Secure storage for MELD dataset and video files
- **Model Artifacts**: Versioned storage of trained models and checkpoints
- **Logging**: TensorBoard logs and training metrics
- **Data Pipeline**: Efficient data loading with S3 integration

### Amazon SageMaker
- **Managed Training**: Distributed training across multiple instances
- **Experiment Tracking**: Built-in experiment management and hyperparameter tuning
- **Model Deployment**: Real-time inference endpoints
- **Notebook Environment**: Jupyter notebooks for development and analysis

## Usage

### Basic Inference
```python
from models import MultimodelSentimentAnalyzer

# Initialize model
model = MultimodelSentimentAnalyzer()
model.eval()

# Prepare inputs
text_inputs = {
    "input_ids": tokenized_text,
    "attention_mask": attention_mask
}
video_frames = video_tensor  # Shape: [batch, frames, channels, height, width]
audio_features = audio_tensor  # Shape: [batch, channels, height, width]

# Run inference
with torch.inference_mode():
    outputs = model(text_inputs, video_frames, audio_features)
    emotion_probs = torch.softmax(outputs["emotion"], dim=1)
    sentiment_probs = torch.softmax(outputs["sentiment"], dim=1)
```

### Training Pipeline
```python
from models import MultiModelTrainer

# Initialize trainer
trainer = MultiModelTrainer(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    gradient_accumulation_steps=2
)

# Train for one epoch
train_losses = trainer.train_epoch()
val_losses, val_metrics = trainer.evaluate(val_loader)
```

## Performance Features

### Training Optimizations
- **Gradient Clipping**: Prevents gradient explosion (max_norm=1.0)
- **Learning Rate Scheduling**: Adaptive learning rate adjustment
- **Early Stopping**: Validation-based training termination
- **Checkpoint Management**: Regular model state saving

### Monitoring & Logging
- **Real-time Metrics**: Loss tracking for both tasks
- **Validation Metrics**: Accuracy and precision for emotion/sentiment
- **TensorBoard Integration**: Visual training progress
- **AWS CloudWatch**: Infrastructure monitoring

## Emotion & Sentiment Mapping

### Emotion Classes
```python
emotion_map = {
    0: "anger", 1: "disgust", 2: "fear", 3: "joy",
    4: "neutral", 5: "sadness", 6: "surprise"
}
```

### Sentiment Classes
```python
sentiment_map = {
    0: "negative", 1: "neutral", 2: "positive"
}
```

## ðŸ“ Dataset Structure

The model expects data in the following format:
- **Text**: Tokenized input with attention masks
- **Video**: Frame sequences [batch, frames, channels, height, width]
- **Audio**: Mel-spectrogram features [batch, channels, frequency, time]
- **Labels**: Integer-encoded emotion and sentiment labels

## Research Applications

This multimodal approach addresses key challenges in video content analysis:
- **Cross-modal Feature Fusion**: Effective integration of heterogeneous data
- **Temporal Modeling**: Capturing temporal dependencies in video sequences  
- **Robust Predictions**: Handling inconsistent single-modality signals
- **Missing Modality**: Graceful degradation when inputs are incomplete

## Contributing

1. Fork the repository
2. Create feature branches for development
3. Ensure comprehensive testing
4. Submit pull requests with detailed descriptions

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **MELD Dataset**: Multimodal EmotionLines Dataset for emotion recognition
- **Pretrained Models**: BERT (Google), R3D-18 (Facebook Research)
- **AWS Services**: EC2, S3, and SageMaker for scalable ML infrastructure

## ðŸ“ž Contact

For questions, issues, or collaboration opportunities, please open an issue on the GitHub repository.
