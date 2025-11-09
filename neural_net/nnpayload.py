import numpy as np
from mnist_model import predict_digit_from_28x28

LED_COUNT = 34  # 6 input + 6 h1 + 6 h2 + 6 h3 + 10 output

LAYER_OFFSETS = {
    "input": (0, 6),    # LEDs 0-5   (row 1: 1A-1F)
    "h1":    (6, 12),   # LEDs 6-11  (row 2: 2A-2F)
    "h2":    (12, 18),  # LEDs 12-17 (row 3: 3A-3F)
    "h3":    (18, 24),  # LEDs 18-23 (row 4: 4A-4F)
    "out":   (24, 34),  # LEDs 24-33 (rows 5-6: 5A-5F, 6A-6D)
}

def compute_input_activations_from_image(x28: np.ndarray) -> np.ndarray:
    """
    Collapse 28x28 image into 6 'input neurons' via 2x3 grid regions.
    Returns: shape (6,) float32.
    """
    if x28.shape != (28, 28):
        raise ValueError(f"Expected 28x28 image, got {x28.shape}")

    # Divide into 2 rows x 3 columns = 6 regions
    # Each region is approximately 14x9 pixels
    h = 14  # half height
    w = 9   # ~third width (28/3 ≈ 9.33)
    
    regions = [
        x28[0:h, 0:w],           # top-left
        x28[0:h, w:2*w],          # top-middle
        x28[0:h, 2*w:28],         # top-right
        x28[h:28, 0:w],           # bottom-left
        x28[h:28, w:2*w],         # bottom-middle
        x28[h:28, 2*w:28],        # bottom-right
    ]
    
    vals = np.array([r.mean() for r in regions], dtype=np.float32)
    return vals

def normalize_to_0_1(v: np.ndarray) -> np.ndarray:
    """
    Scale nonnegative activations so max -> 1.0.
    Returns float32 array with values in [0.0, 1.0].
    """
    v = np.maximum(v, 0.0)
    vmax = float(v.max())
    if vmax == 0.0:
        return np.zeros_like(v, dtype=np.float32)
    v_norm = v / vmax
    return v_norm.astype(np.float32)

def average_to_n_groups(v: np.ndarray, n: int) -> np.ndarray:
    """
    Average values into n groups, handling remainder distribution.
    Input shape (m,) -> Output shape (n,)
    """
    m = len(v)
    if m < n:
        raise ValueError(f"Input length {m} must be >= target groups {n}")
    
    # Calculate group sizes (distribute remainder)
    group_size = m // n
    remainder = m % n
    
    result = np.zeros(n, dtype=v.dtype)
    idx = 0
    
    for i in range(n):
        # First 'remainder' groups get one extra element
        size = group_size + (1 if i < remainder else 0)
        result[i] = v[idx:idx+size].mean()
        idx += size
    
    return result

def compute_layer_brightness(x28: np.ndarray, acts: dict):
    """
    Compute brightness arrays for each layer:
      - 'input': (6,)   from 2x3 grid regions of image
      - 'h1':    (6,)   from acts["hidden1"] (160 neurons averaged to 6)
      - 'h2':    (6,)   from acts["hidden2"] (96 neurons averaged to 6)
      - 'h3':    (6,)   from random numbers (simulated hidden layer)
      - 'out':   (10,)  from acts["output"] (direct mapping)
    Returns dict layer_name -> np.ndarray[float32] with values in [0.0, 1.0].
    """
    input_vals = compute_input_activations_from_image(x28)
    input_b = normalize_to_0_1(input_vals)

    # Hidden layer 1: 160 neurons -> average to 6 LEDs
    h1_raw = normalize_to_0_1(acts["hidden1"])
    h1_b = average_to_n_groups(h1_raw, 6)

    # Hidden layer 2: 96 neurons -> average to 6 LEDs (96/6 = 16 exactly)
    h2_raw = normalize_to_0_1(acts["hidden2"])
    h2_b = average_to_n_groups(h2_raw, 6)

    # Hidden layer 3: Generate random activations (simulated)
    h3_raw = np.random.rand(6).astype(np.float32)
    h3_b = normalize_to_0_1(h3_raw)

    out_b = normalize_to_0_1(acts["output"])

    return {
        "input": input_b,
        "h1":    h1_b,
        "h2":    h2_b,
        "h3":    h3_b,
        "out":   out_b,
    }

def build_led_buffer(x28: np.ndarray):
    """
    Main NN-side API:
      - Takes a 28x28 image
      - Runs NN
      - Returns:
          digit: int (0-9)
          led_buffer: np.ndarray shape (34,), dtype=float32
                      brightness values 0.0-1.0 for each LED in linear order.
                      Use matrix_coordinate_map() to convert to physical matrix positions.
    """
    digit, output, acts = predict_digit_from_28x28(x28)
    layer_b = compute_layer_brightness(x28, acts)

    led_buffer = np.zeros(LED_COUNT, dtype=np.float32)

    # Map layers to linear buffer
    # Input: 6 LEDs (0-5)
    # Physical mapping: 1A, 1B, 1C, 1D, 1E, 1F
    led_buffer[0:6] = layer_b["input"]
    
    # Hidden1: 6 LEDs (6-11)
    # Physical mapping: 2A, 2B, 2C, 2D, 2E, 2F
    led_buffer[6:12] = layer_b["h1"]
    
    # Hidden2: 6 LEDs (12-17)
    # Physical mapping: 3A, 3B, 3C, 3D, 3E, 3F
    led_buffer[12:18] = layer_b["h2"]
    
    # Hidden3: 6 LEDs (18-23)
    # Physical mapping: 4A, 4B, 4C, 4D, 4E, 4F
    led_buffer[18:24] = layer_b["h3"]
    
    # Output: 10 LEDs (24-33)
    # Physical mapping: 5A-5F (digits 0-5), 6A-6D (digits 6-9)
    # Use confidence values (softmax probabilities) directly - already in range 0.0-1.0
    # All 10 digits show their confidence levels, with the predicted digit naturally brightest
    output_confidences = output.astype(np.float32)
    led_buffer[24:34] = output_confidences

    return digit, led_buffer


def create_2d_matrix(led_buffer: np.ndarray) -> np.ndarray:
    """
    Convert linear LED buffer into a 2D array representation of the physical matrix.
    
    Returns a 7x6 numpy array where:
    - Row indices: 0-6 (representing physical rows 1-7)
    - Col indices: 0-5 (representing columns A-F)
    - Values: brightness (0.0-1.0) or 0.0 for empty positions
    - Row 7 (index 6) is empty/faked (all zeros)
    
    Args:
        led_buffer: Linear array of brightness values (34 elements, 0.0-1.0)
    
    Returns:
        2D numpy array shape (7, 6) with brightness values (float32)
    """
    if len(led_buffer) != LED_COUNT:
        raise ValueError(f"Expected buffer length {LED_COUNT}, got {len(led_buffer)}")
    
    # Initialize 2D matrix with zeros (7 rows x 6 cols)
    MATRIX_ROWS = 7
    MATRIX_COLS = 6
    matrix_2d = np.zeros((MATRIX_ROWS, MATRIX_COLS), dtype=np.float32)
    
    # Map each LED from linear buffer to 2D position
    for i, brightness in enumerate(led_buffer):
        row, col = matrix_coordinate_map(i)
        # Convert row (1-6) to index (0-5)
        # Convert col ('A'-'F') to index (0-5)
        row_idx = row - 1
        col_idx = ord(col) - ord('A')
        matrix_2d[row_idx, col_idx] = brightness
    
    # Row 7 (index 6) remains all zeros (faked/empty row)
    
    return matrix_2d


def matrix_coordinate_map(led_index: int) -> tuple[int, str]:
    """
    Convert linear LED index to physical matrix coordinate (row, column).
    
    Physical layout:
        Row 1 (input):    1A-1F  -> indices 0-5
        Row 2 (hidden1):  2A-2F  -> indices 6-11
        Row 3 (hidden2):  3A-3F  -> indices 12-17
        Row 4 (hidden3):  4A-4F  -> indices 18-23
        Row 5 (output):   5A-5F  -> indices 24-29 (digits 0-5)
        Row 6 (output):   6A-6D  -> indices 30-33 (digits 6-9)
    
    Returns: (row: int, col: str) where col is 'A'-'F'
    """
    if led_index < 0 or led_index >= LED_COUNT:
        raise ValueError(f"LED index {led_index} out of range [0, {LED_COUNT})")
    
    if led_index < 6:
        # Row 1: input layer
        return (1, chr(ord('A') + led_index))
    elif led_index < 12:
        # Row 2: hidden1
        return (2, chr(ord('A') + (led_index - 6)))
    elif led_index < 18:
        # Row 3: hidden2
        return (3, chr(ord('A') + (led_index - 12)))
    elif led_index < 24:
        # Row 4: hidden3
        return (4, chr(ord('A') + (led_index - 18)))
    elif led_index < 30:
        # Row 5: output (digits 0-5)
        return (5, chr(ord('A') + (led_index - 24)))
    else:
        # Row 6: output (digits 6-9, only A-D)
        return (6, chr(ord('A') + (led_index - 30)))


def get_matrix_layout() -> dict:
    """
    Get the full matrix layout mapping.
    Returns dict mapping (row, col) -> linear_index
    """
    layout = {}
    for i in range(LED_COUNT):
        row, col = matrix_coordinate_map(i)
        layout[(row, col)] = i
    return layout