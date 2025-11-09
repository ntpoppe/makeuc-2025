import cv2
import numpy as np

PREPROCESS_OUTPUT_SIZE = (28, 28)
PREPROCESS_PADDING_RATIO = 0.2

def preprocess_for_mnist(
    image,
    output_size: tuple[int, int] = PREPROCESS_OUTPUT_SIZE,
    padding_ratio: float = PREPROCESS_PADDING_RATIO,
) -> np.ndarray:
    """Convert finger-drawn digit to MNIST-like 28x28 image."""
    out_w, out_h = output_size

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    img_size = max(gray.shape[0], gray.shape[1])
    
    if img_size > 300:
        block_size = max(11, int(img_size / 20))
        if block_size % 2 == 0:
            block_size += 1
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, 10
        )
    else:
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        white_ratio = np.sum(binary == 255) / binary.size
        if white_ratio > 0.95 or white_ratio < 0.01:
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    img_size = max(gray.shape[0], gray.shape[1])
    kernel_size = 3 if img_size > 200 else 2
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        dark_pixels = np.where(gray < 200)
        if len(dark_pixels[0]) > 0:
            y_coords = dark_pixels[0]
            x_coords = dark_pixels[1]
            x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
            y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
        else:
            return np.zeros((out_h, out_w), dtype=np.uint8)
    else:
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

    padding_x = int((x_max - x_min) * padding_ratio)
    padding_y = int((y_max - y_min) * padding_ratio)

    x_min = max(0, x_min - padding_x)
    y_min = max(0, y_min - padding_y)
    x_max = min(gray.shape[1], x_max + padding_x)
    y_max = min(gray.shape[0], y_max + padding_y)

    roi = gray[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return np.zeros((out_h, out_w), dtype=np.uint8)

    roi = 255 - roi
    roi_h, roi_w = roi.shape

    roi_size = max(roi_w, roi_h)
    if roi_size > 100:
        kernel_size = 3
        iterations = 2
    elif roi_size > 50:
        kernel_size = 2
        iterations = 1
    else:
        kernel_size = 2
        iterations = 1
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    roi = cv2.dilate(roi, kernel, iterations=iterations)

    roi_h, roi_w = roi.shape
    max_dimension = max(roi_w, roi_h)
    if max_dimension > 0:
        scale = 28.0 / max_dimension
        new_w = max(1, int(roi_w * scale))
        new_h = max(1, int(roi_h * scale))
    else:
        new_w, new_h = 1, 1

    resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    output = np.zeros((out_h, out_w), dtype=np.uint8)
    start_x = (out_w - new_w) // 2
    start_y = (out_h - new_h) // 2
    output[start_y:start_y + new_h, start_x:start_x + new_w] = resized

    output = cv2.GaussianBlur(output, (3, 3), 0.5)
    return output


def to_mlp_input(image: np.ndarray) -> np.ndarray:
    """Preprocess image for MLP input: (1, 28, 28), float32, [0, 1]."""
    img = preprocess_for_mnist(image)
    img = img.astype("float32") / 255.0
    return img.reshape(1, 28, 28)


def to_cnn_input(image: np.ndarray) -> np.ndarray:
    """Preprocess image for CNN input: (1, 28, 28, 1), float32, [0, 1]."""
    img = preprocess_for_mnist(image)
    img = img.astype("float32") / 255.0
    return img.reshape(1, 28, 28, 1)
