import numpy as np
import os

if not os.path.exists("mnist_cnn_weights.npz"):
    raise FileNotFoundError(
        "mnist_cnn_weights.npz not found! "
        "Please train the model first: python train_mnist_mlp.py"
    )

weights = np.load("mnist_cnn_weights.npz")

# Load conv layer weights
conv1_kernel = weights["conv1_kernel"]  # (3, 3, 1, 4)
conv1_bias = weights["conv1_bias"]  # (4,)
bn_conv1_gamma = weights["bn_conv1_gamma"]  # (4,)
bn_conv1_beta = weights["bn_conv1_beta"]  # (4,)
bn_conv1_mean = weights["bn_conv1_mean"]  # (4,)
bn_conv1_var = weights["bn_conv1_var"]  # (4,)

# Load dense layer weights
dense1_kernel = weights["dense1_kernel"]  # (64, 676) after transpose (13*13*4=676)
dense1_bias = weights["dense1_bias"]  # (64,)
bn1_gamma = weights["bn1_gamma"]  # (64,)
bn1_beta = weights["bn1_beta"]  # (64,)
bn1_mean = weights["bn1_mean"]  # (64,)
bn1_var = weights["bn1_var"]  # (64,)

dense2_kernel = weights["dense2_kernel"]  # (64, 64) after transpose
dense2_bias = weights["dense2_bias"]  # (64,)
bn2_gamma = weights["bn2_gamma"]  # (64,)
bn2_beta = weights["bn2_beta"]  # (64,)
bn2_mean = weights["bn2_mean"]  # (64,)
bn2_var = weights["bn2_var"]  # (64,)

output_kernel = weights["output_kernel"]  # (10, 64) after transpose
output_bias = weights["output_bias"]  # (10,)

def relu(x):
    return np.maximum(0, x)

def softmax(x, temperature=1.0):
    """Softmax with temperature scaling."""
    x = x / temperature
    x = x - np.max(x)
    x = np.clip(x, -500, 500)
    e = np.exp(x)
    return e / np.sum(e)

def conv2d(x, kernel, bias, stride=1, padding=0):
    """2D convolution operation in NumPy.
    
    Args:
        x: Input tensor of shape (H, W, C_in)
        kernel: Convolution kernel of shape (kernel_h, kernel_w, C_in, C_out)
        bias: Bias of shape (C_out,)
        stride: Stride value
        padding: Padding value (0 for 'valid')
    
    Returns:
        Output tensor of shape (H_out, W_out, C_out)
    """
    kernel_h, kernel_w, in_channels, out_channels = kernel.shape
    h, w, _ = x.shape
    
    # Calculate output dimensions
    h_out = (h - kernel_h) // stride + 1
    w_out = (w - kernel_w) // stride + 1
    
    output = np.zeros((h_out, w_out, out_channels), dtype=x.dtype)
    
    for i in range(h_out):
        for j in range(w_out):
            h_start = i * stride
            h_end = h_start + kernel_h
            w_start = j * stride
            w_end = w_start + kernel_w
            
            # Extract patch
            patch = x[h_start:h_end, w_start:w_end, :]  # (kernel_h, kernel_w, in_channels)
            
            # Convolve with each output channel
            for out_ch in range(out_channels):
                # Element-wise multiply and sum
                output[i, j, out_ch] = np.sum(patch * kernel[:, :, :, out_ch]) + bias[out_ch]
    
    return output

def max_pool2d(x, pool_size=(2, 2), stride=None):
    """2D max pooling operation in NumPy.
    
    Args:
        x: Input tensor of shape (H, W, C)
        pool_size: Pooling window size (pool_h, pool_w)
        stride: Stride value (defaults to pool_size)
    
    Returns:
        Output tensor of shape (H_out, W_out, C)
    """
    if stride is None:
        stride = pool_size
    
    pool_h, pool_w = pool_size
    h, w, c = x.shape
    
    h_out = (h - pool_h) // stride[0] + 1
    w_out = (w - pool_w) // stride[1] + 1
    
    output = np.zeros((h_out, w_out, c), dtype=x.dtype)
    
    for i in range(h_out):
        for j in range(w_out):
            h_start = i * stride[0]
            h_end = h_start + pool_h
            w_start = j * stride[1]
            w_end = w_start + pool_w
            
            patch = x[h_start:h_end, w_start:w_end, :]  # (pool_h, pool_w, c)
            output[i, j, :] = np.max(patch, axis=(0, 1))
    
    return output

def batch_norm(x, gamma, beta, mean, var, epsilon=1e-3):
    """Batch normalization operation.
    
    Args:
        x: Input tensor (can be 1D for dense layers or 3D for conv layers)
        gamma: Scale parameter
        beta: Shift parameter
        mean: Running mean
        var: Running variance
        epsilon: Small constant for numerical stability
    
    Returns:
        Normalized tensor
    """
    # Handle both 1D (dense) and 3D (conv) cases
    if x.ndim == 1:
        # Dense layer: (C,) -> (C,)
        normalized = (x - mean) / np.sqrt(var + epsilon)
        return gamma * normalized + beta
    elif x.ndim == 3:
        # Conv layer: (H, W, C) -> (H, W, C)
        # Reshape to (1, 1, C) for proper broadcasting across spatial dimensions
        mean = mean.reshape(1, 1, -1)
        var = var.reshape(1, 1, -1)
        gamma = gamma.reshape(1, 1, -1)
        beta = beta.reshape(1, 1, -1)
        normalized = (x - mean) / np.sqrt(var + epsilon)
        return gamma * normalized + beta
    else:
        raise ValueError(f"Unsupported tensor dimension for batch_norm: {x.ndim}")

def run_nn_stepwise(x_img: np.ndarray):
    """Run CNN forward pass stepwise.
    
    Args:
        x_img: Input image of shape (28, 28, 1) or (28, 28)
    
    Returns:
        output: Softmax probabilities
        acts: Dictionary of activations at each layer
    """
    acts = {}
    
    # Ensure input is (28, 28, 1)
    if x_img.ndim == 2:
        x_img = np.expand_dims(x_img, axis=-1)
    elif x_img.ndim == 3 and x_img.shape[2] != 1:
        raise ValueError(f"Expected single channel, got {x_img.shape[2]} channels")
    
    acts["input"] = x_img.copy()
    
    # Conv1: (28, 28, 1) -> (26, 26, 4)
    x = conv2d(x_img, conv1_kernel, conv1_bias, stride=1, padding=0)
    x = batch_norm(x, bn_conv1_gamma, bn_conv1_beta, bn_conv1_mean, bn_conv1_var)
    x = relu(x)
    acts["conv1"] = x.copy()
    
    # Pool1: (26, 26, 4) -> (13, 13, 4)
    x = max_pool2d(x, pool_size=(2, 2), stride=(2, 2))
    acts["pool1"] = x.copy()
    
    # Flatten: (13, 13, 4) -> (676,)
    x_flat = x.flatten()
    acts["flatten"] = x_flat.copy()
    
    # Dense1: (676,) -> (64,)
    z1 = dense1_kernel @ x_flat + dense1_bias
    z1 = batch_norm(z1, bn1_gamma, bn1_beta, bn1_mean, bn1_var)
    a1 = relu(z1)
    acts["dense1"] = a1.copy()
    
    # Dense2: (64,) -> (64,)
    z2 = dense2_kernel @ a1 + dense2_bias
    z2 = batch_norm(z2, bn2_gamma, bn2_beta, bn2_mean, bn2_var)
    a2 = relu(z2)
    acts["dense2"] = a2.copy()
    
    # Output: (64,) -> (10,)
    z3 = output_kernel @ a2 + output_bias
    acts["logits"] = z3.copy()
    out = softmax(z3, temperature=1.5)
    acts["output"] = out
    
    return out, acts

def predict_digit_from_28x28(x28: np.ndarray):
    """Predict digit from 28x28 image.
    
    Args:
        x28: Input image of shape (28, 28) or (28, 28, 1), values in [0, 1]
    
    Returns:
        digit: Predicted digit (0-9)
        output: Softmax probabilities
        acts: Dictionary of activations at each layer
    """
    # Ensure values are in [0, 1] range
    if x28.dtype != np.float32:
        x28 = x28.astype(np.float32)
    if x28.max() > 1.0:
        x28 = x28 / 255.0
    
    output, acts = run_nn_stepwise(x28)
    digit = int(np.argmax(output))
    return digit, output, acts
