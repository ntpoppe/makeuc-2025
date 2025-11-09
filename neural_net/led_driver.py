import numpy as np
from gpiozero import PWMLED

# Matrix configuration
MATRIX_ROWS = 7
MATRIX_COLS = 6  # A-F

def handle_led_buffer(matrix_2d: np.ndarray, digit: int | None = None):
    print_2d_matrix(matrix_2d)
    
    if digit is not None:
        print(f"[led_driver] Predicted digit: {digit}")

def print_2d_matrix(matrix_2d: np.ndarray):
    print("[led_driver] Received 2D matrix:")
    print("     A      B      C      D      E      F")
    for row_idx in range(MATRIX_ROWS):
        row_num = row_idx + 1
        row_str = f"{row_num}  "
        for col_idx in range(MATRIX_COLS):
            val = matrix_2d[row_idx, col_idx]
            row_str += f"{val:5.2f} "
        print(row_str)


