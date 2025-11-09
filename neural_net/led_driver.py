import numpy as np
from nnpayload import matrix_coordinate_map, LED_COUNT

# Matrix configuration
MATRIX_ROWS = 6
MATRIX_COLS = 6  # A-F
COL_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F']

# Breadboard addressing
BREADBOARD_CONFIG = {
    # Breadboard 1: Rows 1-3 (input + hidden layers)
    "breadboard_1": {
        "address": 0x00,  # I2C address or other addressing scheme
        "rows": [1, 2, 3],  # Rows 1-3
        "cols": COL_LETTERS,  # All columns A-F
    },
    # Breadboard 2: Rows 4-5 (output layer, only A-B columns)
    "breadboard_2": {
        "address": 0x01,
        "rows": [4, 5],  # Rows 4-5
        "cols": ['A', 'B'],  # Only columns A-B (C-F are empty on rows 4-5)
    },
    # Breadboard 3: Row 6 (output layer, all columns)
    "breadboard_3": {
        "address": 0x02,
        "rows": [6],  # Row 6
        "cols": COL_LETTERS,  # All columns A-F
    },
}


class LEDMatrixDriver:
    """
    Driver for addressing LED matrix across multiple breadboards.
    
    Physical layout:
        Breadboard 1 (rows 1-3):
            Row 1: 1A-1F (input layer, 6 LEDs)
            Row 2: 2A-2F (hidden1, 6 LEDs)
            Row 3: 3A-3F (hidden2, 6 LEDs)
        
        Breadboard 2 (rows 4-5):
            Row 4: 4A-4B (output, 2 LEDs)
            Row 5: 5A-5B (output, 2 LEDs)
        
        Breadboard 3 (row 6):
            Row 6: 6A-6F (output, 6 LEDs)
    """
    
    def __init__(self, breadboard_config: dict = None):
        """
        Initialize driver with breadboard configuration.
        
        Args:
            breadboard_config: Dict mapping breadboard names to configs.
                              If None, uses default BREADBOARD_CONFIG.
        """
        self.config = breadboard_config or BREADBOARD_CONFIG.copy()
        self._validate_config()
    
    def _validate_config(self):
        """Validate that all matrix positions are covered by breadboards."""
        covered = set()
        for name, config in self.config.items():
            for row in config["rows"]:
                for col in config["cols"]:
                    covered.add((row, col))
        
        # Check all required positions
        required = set()
        for i in range(LED_COUNT):
            row, col = matrix_coordinate_map(i)
            required.add((row, col))
        
        missing = required - covered
        if missing:
            raise ValueError(f"Missing matrix positions: {sorted(missing)}")
    
    def get_breadboard_for_position(self, row: int, col: str) -> str:
        """
        Find which breadboard handles a given matrix position.
        
        Returns:
            Breadboard name/ID
        """
        for name, config in self.config.items():
            if row in config["rows"] and col in config["cols"]:
                return name
        raise ValueError(f"No breadboard found for position ({row}, {col})")
    
    def group_by_breadboard(self, led_buffer: np.ndarray) -> dict:
        """
        Group LED buffer values by breadboard.
        
        Returns:
            Dict mapping breadboard name -> dict of {(row, col): brightness}
        """
        if len(led_buffer) != LED_COUNT:
            raise ValueError(f"Expected buffer length {LED_COUNT}, got {len(led_buffer)}")
        
        grouped = {name: {} for name in self.config.keys()}
        
        for i, brightness in enumerate(led_buffer):
            row, col = matrix_coordinate_map(i)
            breadboard = self.get_breadboard_for_position(row, col)
            grouped[breadboard][(row, col)] = brightness
        
        return grouped
    
    def send_to_hardware(self, led_buffer: np.ndarray, digit: int | None = None):
        """
        Send LED buffer to hardware, addressing appropriate breadboards.
        
        This is a sketch - actual implementation depends on hardware protocol:
        - I2C: Use breadboard address to send commands
        - Serial: Send packets with breadboard ID
        - GPIO: Direct pin addressing per breadboard
        
        Args:
            led_buffer: Array of brightness values (0-255) for each LED
            digit: Predicted digit (0-9), optional
        """
        grouped = self.group_by_breadboard(led_buffer)
        
        for breadboard_name, positions in grouped.items():
            config = self.config[breadboard_name]
            address = config["address"]
            
            # TODO: Implement actual hardware communication
            # Example structure:
            #   - Send start command to breadboard at address
            #   - Send (row, col, brightness) tuples for each position
            #   - Send end/commit command
            
            print(f"[led_driver] Breadboard '{breadboard_name}' (addr={hex(address)}):")
            for (row, col), brightness in sorted(positions.items()):
                print(f"  {row}{col}: {brightness}")
        
        if digit is not None:
            print(f"[led_driver] Predicted digit: {digit}")


def handle_led_buffer(led_buffer: np.ndarray, digit: int | None = None):
    """
    Hardware side entrypoint.
    
    Args:
        led_buffer: Array of brightness values (0-255) for each LED
        digit: Predicted digit (0-9), optional
    """
    driver = LEDMatrixDriver()
    driver.send_to_hardware(led_buffer, digit)