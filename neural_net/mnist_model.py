import numpy as np
import os

if not os.path.exists("mnist_mlp_weights.npz"):
    raise FileNotFoundError(
        "mnist_mlp_weights.npz not found! "
        "Please train the model first: python train_mnist_mlp.py"
    )

weights = np.load("mnist_mlp_weights.npz")

expected_shapes = {
    "W1": (20, 784), # First hidden layer: 20 neurons
    "b1": (20,),
    "W2": (12, 20), # Second hidden layer: 12 neurons
    "b2": (12,),
    "W3": (10, 12), # Output layer: 10 neurons
    "b3": (10,),
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
W3 = weights["W3"]
b3 = weights["b3"]

def relu(x):
    return np.maximum(0, x)

def softmax(x, temperature=1.0):
    """Softmax with temperature scaling."""
    x = x / temperature
    x = x - np.max(x)
    x = np.clip(x, -500, 500)
    e = np.exp(x)
    return e / np.sum(e)

def run_nn_stepwise(x_flat: np.ndarray):
    """Run neural network forward pass.
    
    Architecture: 784 → 20 → 12 → 10
    """
    acts = {}
    acts["input"] = x_flat

    # First hidden layer: 784 → 20 (ReLU activation)
    z1 = W1 @ x_flat + b1
    a1 = relu(z1)
    acts["hidden1"] = a1

    # Second hidden layer: 20 → 12 (ReLU activation)
    z2 = W2 @ a1 + b2
    a2 = relu(z2)
    acts["hidden2"] = a2

    # Output layer: 12 → 10 (softmax activation)
    z3 = W3 @ a2 + b3
    acts["logits"] = z3.copy()
    out = softmax(z3, temperature=1.5)
    acts["output"] = out

    return out, acts

def predict_digit_from_28x28(x28: np.ndarray):
    """Predict digit from 28x28 image."""
    x_flat = x28.astype(np.float32).flatten()
    output, acts = run_nn_stepwise(x_flat)
    digit = int(np.argmax(output))
    return digit, output, acts
