"""Configuration settings for finger drawing application."""

# Directory settings
TRACES_DIR = "traces"
TRACES_PREPROCESSED_DIR = "traces_preprocessed"

# Drawing settings
LINE_COLOR = (0, 0, 0)  # Black (BGR)
LINE_THICKNESS = 6
NO_HAND_FRAMES_THRESHOLD = 10

# Preprocessing settings 
PREPROCESS_OUTPUT_SIZE = (28, 28)  # (width, height)
PREPROCESS_PADDING_RATIO = 0.1  # 10% padding around digit

# MediaPipe settings
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6
MAX_NUM_HANDS = 1

# Camera settings
CAMERA_INDEX = 0

