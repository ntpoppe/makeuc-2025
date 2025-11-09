TRACES_DIR = "traces"
TRACES_PREPROCESSED_DIR = "traces_preprocessed"

LINE_COLOR = (0, 0, 0)
LINE_THICKNESS = 8 
NO_HAND_FRAMES_THRESHOLD = 10

PREPROCESS_OUTPUT_SIZE = (28, 28)
PREPROCESS_PADDING_RATIO = 0.1

MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6
MAX_NUM_HANDS = 1

CAMERA_INDEX = 0

SMOOTHING_ALPHA = 0.5 # 0–1; higher = follow finger more, lower = smoother
MIN_MOVE_PIXELS = 2 # ignore tiny movements (jitter)
MAX_JUMP_PIXELS = 80 # treat bigger jumps as “pen up”

MIN_LINE_THICKNESS = 10 # thinnest line (when moving fast)
MAX_LINE_THICKNESS = 20 # thickest line (when moving slow)
SPEED_FOR_MIN_THICK = 600.0 # pixels/sec → MIN thickness (faster = thinner)
SPEED_FOR_MAX_THICK = 80.0 # pixels/sec → MAX thickness (slower = thicker)
THICKNESS_SMOOTHING_ALPHA = 0.3
