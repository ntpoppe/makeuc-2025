# preprocess.py
import cv2
import numpy as np

# Default settings for MNIST-like preprocessing
PREPROCESS_OUTPUT_SIZE = (28, 28)   # (width, height)
PREPROCESS_PADDING_RATIO = 0.2      # 20% padding around bounding box

def preprocess_for_mnist(
    image,
    output_size: tuple[int, int] = PREPROCESS_OUTPUT_SIZE,
    padding_ratio: float = PREPROCESS_PADDING_RATIO,
) -> np.ndarray:
    """
    Convert a finger-drawn digit on a canvas into an MNIST-like 28x28 image.

    Args:
        image: BGR or grayscale image (H x W x 3 or H x W).
        output_size: (width, height) of output image. Default (28, 28).
        padding_ratio: Extra padding as a fraction of bounding-box size.

    Returns:
        output: uint8 array of shape (output_h, output_w),
                with digit bright on dark background (MNIST style).
    """
    out_w, out_h = output_size

    # 1) Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 2) Threshold (invert) just for contour detection:
    #    - THRESH_BINARY_INV: dark strokes become white (255), bg becomes black (0)
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Mild cleanup to remove noise
    kernel_small = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)

    # 3) Find contours of the digit
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        # No digit -> return a blank MNIST-like image (background only)
        return np.zeros((out_h, out_w), dtype=np.uint8)

    # 4) Bounding box around all strokes
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

    # 5) Add padding around digit
    padding_x = int((x_max - x_min) * padding_ratio)
    padding_y = int((y_max - y_min) * padding_ratio)

    x_min = max(0, x_min - padding_x)
    y_min = max(0, y_min - padding_y)
    x_max = min(gray.shape[1], x_max + padding_x)
    y_max = min(gray.shape[0], y_max + padding_y)

    roi = gray[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return np.zeros((out_h, out_w), dtype=np.uint8)

    # 6) Invert intensity to match MNIST:
    #    - Input: dark digit on light bg
    #    - MNIST: bright digit on dark bg
    roi = 255 - roi

    # Optional: thicken strokes slightly so they survive downscaling
    kernel = np.ones((2, 2), np.uint8)
    roi = cv2.dilate(roi, kernel, iterations=1)

    # 7) Resize ROI: longest side -> 20 pixels, keep aspect ratio
    roi_h, roi_w = roi.shape
    scale = 20.0 / max(roi_w, roi_h)
    new_w = max(1, int(roi_w * scale))
    new_h = max(1, int(roi_h * scale))

    resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 8) Center inside a 28x28 canvas
    output = np.zeros((out_h, out_w), dtype=np.uint8)  # background=0
    start_x = (out_w - new_w) // 2
    start_y = (out_h - new_h) // 2
    output[start_y:start_y + new_h, start_x:start_x + new_w] = resized

    # 9) Light Gaussian blur to mimic MNIST anti-aliasing
    output = cv2.GaussianBlur(output, (3, 3), 0.5)

    return output


def to_mlp_input(image: np.ndarray) -> np.ndarray:
    """
    Preprocess an image and return data shaped for your MLP:
        model: Input(shape=(28, 28))
    Output shape: (1, 28, 28), float32, range [0, 1]
    """
    img = preprocess_for_mnist(image)
    img = img.astype("float32") / 255.0
    return img.reshape(1, 28, 28)


def to_cnn_input(image: np.ndarray) -> np.ndarray:
    """
    Preprocess an image and return data shaped for a CNN:
        model: Input(shape=(28, 28, 1))
    Output shape: (1, 28, 28, 1), float32, range [0, 1]
    """
    img = preprocess_for_mnist(image)
    img = img.astype("float32") / 255.0
    return img.reshape(1, 28, 28, 1)
