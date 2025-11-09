import numpy as np

def handle_led_buffer(led_buffer: np.ndarray, digit: int | None = None):
    """
    Hardware side entrypoint.
    """
    # Placeholder
    print(f"[led_driver] digit={digit}, first 10 values={led_buffer[:10]}")