# Neural Network Digit Recognition with LED Visualization

A real-time digit recognition system that captures finger-drawn digits via camera, recognizes them using a neural network, and visualizes the network's internal activations through LED displays.

## Overview

This project demonstrates an end-to-end machine learning pipeline:
- **Input**: Finger drawing captured via camera using MediaPipe hand tracking
- **Processing**: Preprocessing pipeline converts drawings to MNIST-compatible format
- **Recognition**: Multi-layer perceptron (MLP) neural network predicts digits (0-9)
- **Visualization**: LED array displays neural network activations in real-time

##  Architecture

### Neural Network Architecture
- **Input Layer**: 784 neurons (28×28 image flattened)
- **Hidden Layer 1**: 160 neurons (ReLU activation)
- **Hidden Layer 2**: 96 neurons (ReLU activation)
- **Output Layer**: 10 neurons (softmax activation, one per digit)

**Note**: For LED display, hidden layers are averaged down to 20 and 12 LEDs respectively (groups of 8).

### System Flow
Below is a diagram of the intended workflow for the system.

```
Camera Frame → MediaPipe Hand Detection → Gesture Recognition
    ↓
Fingertip Tracking → Canvas Drawing
    ↓
Canvas Image → Preprocessing Pipeline → 28×28 MNIST Format
    ↓
28×28 Image → Neural Network Forward Pass → Digit Prediction + Activations
    ↓
Activations → LED Buffer Generation (46 values) → Hardware Driver
    ↓
LED Display
```

### LED Buffer Structure (46 LEDs)

- **Input layer (LEDs 0-3)**: 4 LEDs representing image quadrants (else would need 784 LEDs)
- **Hidden Layer 1 (LEDs 4-23)**: 20 LEDs showing averaged activations from 160 neurons
- **Hidden Layer 2 (LEDs 24-35)**: 12 LEDs showing averaged activations from 96 neurons
- **Output layer (LEDs 36-45)**: 10 LEDs showing digit probabilities (0-9)

## Project Structure

```
neural_net/
├── finger_draw.py          # Main application - camera capture & drawing
├── preprocessing.py        # Image preprocessing for MNIST compatibility
├── mnist_model.py          # Neural network model definition & inference
├── train_mnist_mlp.py      # Model training script
├── nnpayload.py            # LED buffer generation from activations
├── led_driver.py           # Hardware interface for LED control
├── config.py               # Configuration parameters
├── requirements.txt        # Python dependencies
└── mnist_mlp_weights.npz   # Trained model weights (generated after training)
```

## Setup

### Prerequisites

- Python 3.8+
- Camera/webcam

1. Clone the repo!

2. Initialize the virtual environment and install dependencies
   ```bash
   cd neural_net
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Train the model** (if weights file doesn't exist, but I commited them haha)
   ```bash
   python train_mnist_mlp.py
   ```

## Usage

### Running the Application

```bash
cd neural_net
python finger_draw.py
```

### Controls

- **Draw**: Extend index and middle fingers together to draw
- **Clear**: Press `c` to clear the canvas
- **Predict**: Press `p` to manually trigger prediction
- **Quit**: Press `q` to exit

### How It Works

1. **Drawing**: The system tracks your hand and draws on a virtual canvas when two fingers are extended
2. **Auto-Prediction**: When you remove your hand (after a threshold of frames), the system automatically:
   - Preprocesses the drawing
   - Runs neural network inference
   - Displays the predicted digit and confidence
   - Generates LED buffer for visualization (if available)
3. **Visualization**: The LED buffer represents neural network activations, making the decision process visible

## Machine Learning
The model is trained on the MNIST dataset with the following architecture:

- **Architecture**: 784 → 160 → 96 → 10
- **Activation**: ReLU for hidden layers, Softmax for output
- **Optimizer**: Adam with exponential decay
- **Regularization**: Early stopping and model checkpointing

To retrain with different architectures, modify `train_mnist_mlp.py` and ensure `mnist_model.py` and `nnpayload.py` match the new architecture.

## 🎨 Features

- **Real-time processing**: ~30 FPS camera processing with hand tracking
- **Robust preprocessing**: Handles varied drawing styles and lighting conditions
- **Interpretability**: LED visualization shows network activations at each layer
- **Modular design**: Separate modules for capture, preprocessing, inference, visualization, and hardware
- **Cross-platform camera support**: GStreamer, V4L2, and generic backends

## 📊 Technical Details

### Preprocessing Pipeline

1. **Binarization**: Adaptive thresholding or Otsu thresholding
2. **Morphology**: Closing/opening operations to clean noise
3. **Bounding Box**: Finds digit region with padding
4. **Normalization**: Centers and scales to 28×28
5. **Blur & Threshold**: Light smoothing then binary threshold for solid white output

### LED Visualization

The system maps neural network activations to LED brightness:
- Each layer's activations are normalized to 0-255
- Hidden layers are averaged (groups of 8) to fit LED hardware constraints
- Output layer directly shows digit probabilities

## 🤝 Contributing

This project was developed for MakeUC 2025. Contributions welcome!

## 📝 License

[Add your license here]

## 🙏 Acknowledgments

- MNIST dataset
- MediaPipe for hand tracking
- TensorFlow/Keras for neural network training

