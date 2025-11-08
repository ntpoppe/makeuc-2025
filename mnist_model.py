# mnist_model.py
import numpy as np
import os

# Load trained weights
if not os.path.exists("mnist_mlp_weights.npz"):
    raise FileNotFoundError(
        "mnist_mlp_weights.npz not found! "
        "Please train the model first: python train_mnist_mlp.py"
    )

weights = np.load("mnist_mlp_weights.npz")

# Expected dimensions for 784 -> 128 -> 128 -> 64 -> 10
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

# Validate weights match expected architecture
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

W1 = weights["W1"]  # (128,784)
b1 = weights["b1"]  # (128,)
W2 = weights["W2"]  # (128,128)
b2 = weights["b2"]  # (128,)
W3 = weights["W3"]  # (64,128)
b3 = weights["b3"]  # (64,)
W4 = weights["W4"]  # (10,64)
b4 = weights["b4"]  # (10,)

def relu(x):
    return np.maximum(0, x)

def softmax(x, temperature=1.0):
    """Softmax with temperature scaling for better calibration."""
    x = x / temperature
    x = x - np.max(x)
    x = np.clip(x, -500, 500)
    e = np.exp(x)
    return e / np.sum(e)

def run_nn_stepwise(x_flat: np.ndarray):
    """
    x_flat: shape (784,), values in [0,1]
    returns:
        output_probs: (10,)
        acts: dict with 'input', 'hidden1', 'hidden2', 'hidden3', 'output'
    """
    acts = {}

    acts["input"] = x_flat

    z1 = W1 @ x_flat + b1         # (128,)
    a1 = relu(z1)
    acts["hidden1"] = a1

    z2 = W2 @ a1 + b2             # (128,)
    a2 = relu(z2)
    acts["hidden2"] = a2

    z3 = W3 @ a2 + b3             # (64,)
    a3 = relu(z3)
    acts["hidden3"] = a3

    z4 = W4 @ a3 + b4             # (10,)
    acts["logits"] = z4.copy()
    out = softmax(z4, temperature=1.5)
    acts["output"] = out

    return out, acts

def predict_digit_from_28x28(x28: np.ndarray):
    """
    x28: (28,28) float32 or float64, ideally in [0,1]
    returns: (predicted_digit, probs_vector)
    """
    x_flat = x28.astype(np.float32).flatten()  # (784,)
    output, acts = run_nn_stepwise(x_flat)
    digit = int(np.argmax(output))
    return digit, output, acts
