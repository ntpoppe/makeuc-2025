import numpy as np
from mnist_model import predict_digit_from_28x28

LED_COUNT = 46

LAYER_OFFSETS = {
    "input": (0, 4),    # LEDs 0-3
    "h1":    (4, 24),   # LEDs 4-23  (20 neurons)
    "h2":    (24, 36),  # LEDs 24-35 (12 neurons)
    "out":   (36, 46),  # LEDs 36-45 (10 neurons, digits 0-9)
}

def compute_input_activations_from_image(x28: np.ndarray) -> np.ndarray:
    """
    Collapse 28x28 image into 4 'input neurons' via quadrant means.
    Returns: shape (4,) float32.
    """
    if x28.shape != (28, 28):
        raise ValueError(f"Expected 28x28 image, got {x28.shape}")

    q00 = x28[0:14, 0:14]     # top-left
    q01 = x28[0:14, 14:28]    # top-right
    q10 = x28[14:28, 0:14]    # bottom-left
    q11 = x28[14:28, 14:28]   # bottom-right

    vals = np.array(
        [q00.mean(), q01.mean(), q10.mean(), q11.mean()],
        dtype=np.float32,
    )
    return vals

def normalize_to_0_255(v: np.ndarray) -> np.ndarray:
    """
    Scale nonnegative activations so max -> 255.
    """
    v = np.maximum(v, 0.0)
    vmax = float(v.max())
    if vmax == 0.0:
        return np.zeros_like(v, dtype=np.uint8)
    v_norm = v / vmax
    return (v_norm * 255).astype(np.uint8)

def average_groups_of_8(v: np.ndarray) -> np.ndarray:
    """
    Average groups of 8 values: [a, b, c, d, e, f, g, h, ...] -> [(a+b+c+d+e+f+g+h)/8, ...]
    Input shape (n,) -> Output shape (n//8,)
    """
    if len(v) % 8 != 0:
        raise ValueError(f"Input length must be divisible by 8, got {len(v)}")
    reshaped = v.reshape(-1, 8)
    return reshaped.mean(axis=1)

def compute_layer_brightness(x28: np.ndarray, acts: dict):
    """
    Compute brightness arrays for each layer:
      - 'input': (4,)   from quadrants of image
      - 'h1':    (20,)  from acts["hidden1"] (160 neurons averaged to 20, groups of 8)
      - 'h2':    (12,)  from acts["hidden2"] (96 neurons averaged to 12, groups of 8)
      - 'out':   (10,)  from acts["output"]
    Returns dict layer_name -> np.ndarray[uint8].
    """
    input_vals = compute_input_activations_from_image(x28)
    input_b = normalize_to_0_255(input_vals)

    # Hidden layer 1: 160 neurons -> average groups of 8 -> 20 LEDs
    h1_raw = normalize_to_0_255(acts["hidden1"])
    h1_b = average_groups_of_8(h1_raw)

    # Hidden layer 2: 96 neurons -> average groups of 8 -> 12 LEDs
    h2_raw = normalize_to_0_255(acts["hidden2"])
    h2_b = average_groups_of_8(h2_raw)

    out_b = normalize_to_0_255(acts["output"])

    return {
        "input": input_b,
        "h1":    h1_b,
        "h2":    h2_b,
        "out":   out_b,
    }

def build_led_buffer(x28: np.ndarray):
    """
    Main NN-side API:
      - Takes a 28x28 image
      - Runs NN
      - Returns:
          digit: int (0-9)
          led_buffer: np.ndarray shape (46,), dtype=uint8
                      brightness values 0-255 for each 'neuron LED'.
    """
    digit, output, acts = predict_digit_from_28x28(x28)
    layer_b = compute_layer_brightness(x28, acts)

    led_buffer = np.zeros(LED_COUNT, dtype=np.uint8)

    for layer_name, (start, end) in LAYER_OFFSETS.items():
        vals = layer_b[layer_name]
        if end - start != len(vals):
            raise ValueError(
                f"Length mismatch for {layer_name}: "
                f"LED span {start}-{end}, vals len {len(vals)}"
            )
        led_buffer[start:end] = vals

    return digit, led_buffer