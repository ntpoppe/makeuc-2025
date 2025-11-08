import cv2
import numpy as np

from config import PREPROCESS_OUTPUT_SIZE, PREPROCESS_PADDING_RATIO


def preprocess_for_mnist(image, output_size=PREPROCESS_OUTPUT_SIZE, padding_ratio=PREPROCESS_PADDING_RATIO):
    """
    Preprocess image to match MNIST format.
    
    Args:
        image: Input image (BGR or grayscale)
        output_size: Target size (width, height)
        padding_ratio: Padding ratio around digit
    
    Returns:
        Preprocessed grayscale image
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Find bounding box of drawing
    _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.ones(output_size[::-1], dtype=np.uint8) * 255
    
    # Get bounding box of all contours
    x_min = gray.shape[1]
    y_min = gray.shape[0]
    x_max = 0
    y_max = 0
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    
    # Add padding
    padding_x = int((x_max - x_min) * padding_ratio)
    padding_y = int((y_max - y_min) * padding_ratio)
    
    x_min = max(0, x_min - padding_x)
    y_min = max(0, y_min - padding_y)
    x_max = min(gray.shape[1], x_max + padding_x)
    y_max = min(gray.shape[0], y_max + padding_y)
    
    # Extract ROI
    roi = gray[y_min:y_max, x_min:x_max]
    
    if roi.size == 0:
        return np.ones(output_size[::-1], dtype=np.uint8) * 255
    
    # Resize maintaining aspect ratio
    roi_h, roi_w = roi.shape
    output_w, output_h = output_size
    scale = min(output_w / roi_w, output_h / roi_h)
    new_w = int(roi_w * scale)
    new_h = int(roi_h * scale)
    resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Center on white background
    output = np.ones((output_h, output_w), dtype=np.uint8) * 255
    start_x = (output_w - new_w) // 2
    start_y = (output_h - new_h) // 2
    output[start_y:start_y + new_h, start_x:start_x + new_w] = resized
    
    return output

