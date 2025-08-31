#  Hand Gesture Recognition System

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue.svg)](https://www.kaggle.com/gti-upm/leapgestrecog)

> **Task 4** - Prodigy InfoTech Machine Learning Internship

A deep learning-powered hand gesture recognition system that accurately identifies and classifies different hand gestures from image data, enabling intuitive human-computer interaction and gesture-based control systems.


## 🎯 Project Overview

This project develops a Convolutional Neural Network (CNN) model capable of recognizing various hand gestures from the LeapGestRecog dataset. The system processes images of hand gestures and classifies them into different categories, making it suitable for applications in human-computer interaction, accessibility tools, and gesture-based control systems.

### ✨ Key Features

- **Deep CNN Architecture**: Multi-layer convolutional neural network with batch normalization and dropout
- **Data Augmentation**: Enhanced training with rotation, shifting, and flipping transformations
- **High Accuracy**: Achieves robust classification performance across multiple gesture classes
- **Real-time Prediction**: Optimized for fast inference on new gesture images
- **Comprehensive Evaluation**: Detailed performance metrics and confusion matrix analysis
- **Model Persistence**: Save and load trained models for deployment

## 🛠️ Technology Stack

- **Deep Learning**: TensorFlow 2.x, Keras
- **Computer Vision**: OpenCV, PIL
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Model Evaluation**: Scikit-learn
- **Development Environment**: Google Colab

## 📊 Dataset

The project uses the **LeapGestRecog** dataset from Kaggle, which contains:

- **Source**: [Kaggle - LeapGestRecog Dataset](https://www.kaggle.com/gti-upm/leapgestrecog)
- **Images**: Thousands of hand gesture images across multiple classes
- **Format**: RGB images in various resolutions
- **Classes**: Multiple gesture categories (00, 01, 02, etc.)
- **Structure**: Organized in nested folder hierarchy

### Dataset Structure
```
leapGestRecog/
├── 00/
│   ├── 01_palm/
│   ├── 02_l/
│   └── ...
├── 01/
├── 02/
└── ...
```

## 🏗️ Model Architecture

The CNN model features a sophisticated architecture optimized for gesture recognition:

```python
Model Structure:
├── Conv2D(32) + BatchNorm + MaxPool + Dropout(0.25)
├── Conv2D(64) + BatchNorm + MaxPool + Dropout(0.25)
├── Conv2D(128) + BatchNorm + MaxPool + Dropout(0.25)
├── Conv2D(256) + BatchNorm + MaxPool + Dropout(0.25)
├── Flatten
├── Dense(512) + BatchNorm + Dropout(0.5)
├── Dense(256) + BatchNorm + Dropout(0.5)
└── Dense(num_classes, softmax)
```

### Key Components:
- **Convolutional Layers**: Extract hierarchical features from gesture images
- **Batch Normalization**: Stabilize training and improve convergence
- **Dropout Regularization**: Prevent overfitting with strategic dropout layers
- **Data Augmentation**: Enhance model generalization with image transformations

## 🚀 Getting Started

### Prerequisites

```sh
pip install tensorflow keras matplotlib seaborn scikit-learn opencv-python pillow
```

### Installation

1. **Clone the repository**

```sh
   git clone https://github.com/yourusername/hand-gesture-recognition.git
   cd hand-gesture-recognition
```

2. **Download the dataset**
   - Visit [Kaggle LeapGestRecog Dataset](https://www.kaggle.com/gti-upm/leapgestrecog)
   - Download and extract to your preferred location
   - Update the `dataset_path` in the code

3. **Setup Google Drive** (if using Colab)

```python
   from google.colab import drive
   drive.mount('/content/drive')
```

### Usage

#### Training the Model

```python
# Load and preprocess data
X, y, class_names = load_gesture_data(dataset_path)

# Create and compile model
model = create_gesture_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train with callbacks
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=32),
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=callbacks
)
```

#### Making Predictions

```python
# Load trained model
model, class_names = load_trained_model(model_path, class_names_path)

# Predict single gesture
gesture, confidence = predict_gesture(model, image_path, class_names)
print(f"Predicted: {gesture} (Confidence: {confidence:.2f})")
```

#### Real-time Prediction

```python
# Create predictor function
predictor = create_gesture_predictor()

# Predict from image array
result = predictor(image_array)
print(f"Gesture: {result['gesture']}")
print(f"Confidence: {result['confidence']:.4f}")
```

## 📈 Model Performance

### Training Results
- **Training Accuracy**: ~95%+
- **Validation Accuracy**: ~92%+
- **Test Accuracy**: ~90%+
- **Training Time**: ~2-3 hours (50 epochs)

### Evaluation Metrics
- **Classification Report**: Precision, Recall, F1-Score for each gesture class
- **Confusion Matrix**: Visual representation of classification performance
- **Loss Curves**: Training and validation loss progression

## 📁 Project Structure

```
hand-gesture-recognition/
├── README.md
├── requirements.txt
├── gesture_recognition.ipynb
├── models/
│   ├── gesture_recognition_model.h5
│   └── class_names.pkl
├── data/
│   └── leapGestRecog/
├── results/
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── sample_predictions.png
└── utils/
    ├── data_loader.py
    ├── model_builder.py
    └── predictor.py
```

## 🔧 Configuration Options

### Model Hyperparameters

```python
IMG_SIZE = (64, 64)          # Input image dimensions
BATCH_SIZE = 32              # Training batch size
EPOCHS = 50                  # Training epochs
LEARNING_RATE = 0.001        # Initial learning rate
DROPOUT_RATE = 0.5           # Dropout probability
```

### Data Augmentation Settings

```python
ROTATION_RANGE = 20          # Random rotation degrees
WIDTH_SHIFT_RANGE = 0.1      # Horizontal shift fraction
HEIGHT_SHIFT_RANGE = 0.1     # Vertical shift fraction
ZOOM_RANGE = 0.1             # Zoom range
HORIZONTAL_FLIP = True       # Enable horizontal flipping
```

## 🤝 Applications

This gesture recognition system can be applied in various domains:

- **Human-Computer Interaction**: Navigate interfaces with hand gestures
- **Accessibility Tools**: Assist users with motor disabilities
- **Gaming**: Gesture-based game controls
- **Smart Home**: Control IoT devices with gestures
- **Sign Language Recognition**: Foundation for sign language interpretation
- **Augmented Reality**: Natural interaction in AR environments

## 📋 Future Enhancements

- [ ] **Real-time Video Processing**: Extend to video stream gesture recognition
- [ ] **3D Hand Pose Estimation**: Incorporate depth information
- [ ] **Multi-hand Detection**: Recognize gestures from multiple hands
- [ ] **Transfer Learning**: Fine-tune on specialized gesture datasets
- [ ] **Mobile Deployment**: Optimize model for mobile applications
- [ ] **Gesture Sequences**: Recognize dynamic gesture sequences

## 🐛 Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce batch size or image resolution
   - Use data generators for large datasets

2. **Low Accuracy**
   - Increase training epochs
   - Adjust learning rate
   - Add more data augmentation

3. **Overfitting**
   - Increase dropout rates
   - Add more regularization
   - Use early stopping

## 📚 Resources

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Guide](https://keras.io/guides/)
- [OpenCV Tutorials](https://opencv.org/courses/)
- [LeapGestRecog Dataset](https://www.kaggle.com/gti-upm/leapgestrecog)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Prodigy InfoTech** for the internship opportunity
- **Kaggle** for providing the LeapGestRecog dataset
- **GTI-UPM** for creating and maintaining the dataset
- **TensorFlow/Keras** community for excellent documentation

## 👨‍💻 Author

**Suman Banerjee**
- GitHub: [@SumanBanerjee21](https://github.com/SumanBanerjee21)
- LinkedIn: [Suman Banerjee](https://www.linkedin.com/in/suman-banerjee-394822261/)
- Email: indsumanttt2002@gmail.com

---

<div align="center">
  <sub>Built with ❤️ during Machine Learning Internship at Prodigy InfoTech</sub>
</div>