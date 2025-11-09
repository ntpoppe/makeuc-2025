import numpy as np
import os

if not os.path.exists("mnist_mlp_weights.npz"):
    raise FileNotFoundError(
        "mnist_mlp_weights.npz not found! "
        "Please train the model first: python train_mnist_mlp.py"
    )

weights = np.load("mnist_mlp_weights.npz")

expected_shapes = {
    "W1": (128, 784),
    "b1": (128,),
    "W2": (128, 128),
    "b2": (128,),
    "W3": (64, 128),
    "b3": (64,),
    "W4": (10, 64),
    "b4": (10,),
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
W4 = weights["W4"]
b4 = weights["b4"]

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
    """Run neural network forward pass."""
    acts = {}
    acts["input"] = x_flat

    z1 = W1 @ x_flat + b1
    a1 = relu(z1)
    acts["hidden1"] = a1

    z2 = W2 @ a1 + b2
    a2 = relu(z2)
    acts["hidden2"] = a2

    z3 = W3 @ a2 + b3
    a3 = relu(z3)
    acts["hidden3"] = a3

    z4 = W4 @ a3 + b4
    acts["logits"] = z4.copy()
    out = softmax(z4, temperature=1.5)
    acts["output"] = out

    return out, acts

def predict_digit_from_28x28(x28: np.ndarray):
    """Predict digit from 28x28 image."""
    x_flat = x28.astype(np.float32).flatten()
    output, acts = run_nn_stepwise(x_flat)
    digit = int(np.argmax(output))
    return digit, output, acts
