Hand Gesture Recognition (Task 4, Prodigy InfoTech)
A deep learning project to recognize hand gestures from images using a CNN trained on the LeapGestRecog-style dataset. Built with TensorFlow/Keras in Google Colab,
the model achieves strong multi-class accuracy and includes a ready-to-use prediction utility for inference.

Highlights
10-class gesture recognition on ~20K images (64x64 RGB), auto-loaded from nested folders.

CNN with BatchNorm, Dropout, and data augmentation (rotation, shifts, shear, zoom, horizontal flip).

Validation accuracy peak ~97.8% and test accuracy ~97.98% on held-out test set.

Saved model and class names for quick re-use; includes a simple prediction function that returns class + confidence.

Dataset
Expected directory structure (example):
/content/drive/MyDrive/leapGestRecog/
├── 00/
├── 01/
├── ...
└── 09/

The loader recursively reads image files from each class folder and resizes each to 64x64 (RGB, normalized).

Similar to LeapGestRecog (≈20K frames across 10 classes) used widely for gesture recognition tutorials.

Tech Stack
Python 3.12, TensorFlow/Keras, NumPy, Pandas, OpenCV, scikit-learn, Matplotlib, Seaborn (Colab environment)

Model Architecture
4× Conv2D blocks: [Conv → BatchNorm → MaxPool → Dropout] with filters 32, 64, 128, 256.

Flatten → Dense(512) → Dense(256) → Dense(10 softmax) with BatchNorm + Dropout in between.

Optimizer: Adam; Loss: categorical_crossentropy; Metrics: accuracy.

Training Details
Split: 70% train / 15% val / 15% test (via two-stage split).

Augmentation: rotation=20°, shifts=0.1, shear=0.1, zoom=0.1, horizontal_flip=True.

Callbacks: EarlyStopping (monitor val_accuracy), ReduceLROnPlateau (monitor val_loss), ModelCheckpoint (best val_accuracy).

Performance (example run)
Validation accuracy peak: ~97.8% (Epoch 31), Test accuracy: ~97.98%, Test loss: ~0.053.

Classification report shows near-perfect precision/recall/F1 across most classes.

Confusion matrix plotted for per-class error analysis.

Repository Structure (suggested)
Task4.ipynb # Main training and evaluation notebook

/models

gesture_recognition_model.h5 # Saved model (HDF5)

best_gesture_model.h5 # Checkpoint with best val_accuracy

class_names.pkl # Saved class names

/data

leapGestRecog/ # Dataset root (10 class folders)

/assets

plots/ # Accuracy/Loss curves, confusion matrix

Setup & Usage (Colab)
Mount Drive + Install dependencies

The notebook already mounts Drive and installs required packages. Ensure dataset_path points to the correct folder.

Load data

The loader reads nested folders, resizes to 64×64, normalizes to , and builds numpy arrays X (images), y (labels).

Train

Run the training cell to start model.fit with augmentation and callbacks.

Evaluate

The notebook evaluates on the test set, prints a classification report, and renders a confusion matrix.

Save & Load

Trained model is saved to /content/drive/MyDrive/gesture_recognition_model.h5 and class names to class_names.pkl.

Load helper:
model, class_names = load_trained_model('/content/drive/MyDrive/gesture_recognition_model.h5',
'/content/drive/MyDrive/class_names.pkl')

Predict (array-based)

Use the provided factory to create a predictor and pass a preprocessed image array:
predictor = create_gesture_predictor()
result = predictor(image_array) # returns gesture, confidence, and all class probabilities

Notes & Tips
HDF5 is used for model saving in this run; Keras recommends the native “.keras” format. You can switch to model.save('model.keras').

For speed on CPU-only Colab sessions, reduce image size or depth, or use tf.data pipelines.

For real-time webcam inference, add OpenCV capture and apply the same preprocessing (resize 64×64, normalize).

