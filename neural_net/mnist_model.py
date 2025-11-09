import numpy as np
import os

if not os.path.exists("mnist_mlp_weights.npz"):
    raise FileNotFoundError(
        "mnist_mlp_weights.npz not found! "
        "Please train the model first: python train_mnist_mlp.py"
    )

weights = np.load("mnist_mlp_weights.npz")

expected_shapes = {
    "W1": (32, 784), # Hidden layer: 32 neurons
    "b1": (32,),
    "W2": (10, 32), # Output layer: 10 neurons
    "b2": (10,),
}

for key, expected_shape in expected_shapes.items():
    if key not in weights:
        raise ValueError(f"Missing weight: {key}")
    actual_shape = weights[key].shape
    if actual_shape != expected_shape:
        raise ValueError(
            f"Weight dimension mismatch!\n"
            f"  {key}: got {actual_shape}, expected {expected_shape}\n"
            f"  The weights file doesn't match the model architecture.\n"
            f"  Please retrain: python train_mnist_mlp.py"
        )

W1 = weights["W1"]
b1 = weights["b1"]
W2 = weights["W2"]
b2 = weights["b2"]

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU activation function."""
    return np.maximum(alpha * x, x)

def softmax(x, temperature=1.0):
    """Softmax with temperature scaling."""
    x = x / temperature
    x = x - np.max(x)
    x = np.clip(x, -500, 500)
    e = np.exp(x)
    return e / np.sum(e)

def run_nn_stepwise(x_flat: np.ndarray):
    """Run neural network forward pass.
    
    Architecture: 784 → 32 → 10
    """
    acts = {}
    acts["input"] = x_flat

    # Hidden layer: 784 → 32 (Leaky ReLU activation)
    z1 = W1 @ x_flat + b1
    a1 = leaky_relu(z1)
    acts["hidden1"] = a1

    # Output layer: 32 → 10 (softmax activation)
    z2 = W2 @ a1 + b2
    acts["logits"] = z2.copy()
    out = softmax(z2, temperature=1.5)
    acts["output"] = out

    return out, acts

def predict_digit_from_28x28(x28: np.ndarray):
    """Predict digit from 28x28 image."""
    x_flat = x28.astype(np.float32).flatten()
    output, acts = run_nn_stepwise(x_flat)
    digit = int(np.argmax(output))
    return digit, output, acts
