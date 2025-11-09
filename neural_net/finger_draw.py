import os
import cv2
import numpy as np
import mediapipe as mp
import math
import time
from datetime import datetime

from config import (
    TRACES_DIR,
    TRACES_PREPROCESSED_DIR,
    LINE_COLOR,
    LINE_THICKNESS,
    NO_HAND_FRAMES_THRESHOLD,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MAX_NUM_HANDS,
    CAMERA_INDEX,
    SMOOTHING_ALPHA,
    MIN_MOVE_PIXELS,
    MAX_JUMP_PIXELS,
    MIN_LINE_THICKNESS,
    MAX_LINE_THICKNESS,
    SPEED_FOR_MIN_THICK,
    SPEED_FOR_MAX_THICK,
    THICKNESS_SMOOTHING_ALPHA,
)
from preprocessing import preprocess_for_mnist
from mnist_model import predict_digit_from_28x28

def open_camera(index: int = CAMERA_INDEX, width: int = 640, height: int = 480, fps: int = 30):
    """
    Returns a cv2.VideoCapture that works on both headless CLI and VNC.
    Tries (in order):
      1. GStreamer + libcamera
      2. V4L2 device node 
      3. Fallback to any index
    """
    # GStreamer + libcamera
    gst_pipeline = (
        f'libcamerasrc ! '
        f'video/x-raw,width={width},height={height},framerate={fps}/1 ! '
        f'videoconvert ! appsink'
    )
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print("Camera opened with GStreamer (libcamera)")
        return cap

    # Classic V4L2 
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if cap.isOpened():
        print(f"Camera opened with V4L2 (index {index})")
        return cap

    # Plain index
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"Camera opened with generic backend (index {index})")
        return cap

    raise RuntimeError(f"Could not open camera (index {index}) - tried GStreamer, V4L2 and generic")

def smooth_point(prev_point, new_point, alpha=SMOOTHING_ALPHA):
    """Exponential smoothing between previous and new point."""
    if prev_point is None:
        return new_point
    x = int(alpha * new_point[0] + (1 - alpha) * prev_point[0])
    y = int(alpha * new_point[1] + (1 - alpha) * prev_point[1])
    return (x, y)

def detect_two_fingers_up(hand_landmarks, frame_w, frame_h):
    """check if index and middle fingers are extended."""
    index_tip = hand_landmarks.landmark[8]
    index_pip = hand_landmarks.landmark[6]
    middle_tip = hand_landmarks.landmark[12]
    middle_pip = hand_landmarks.landmark[10]
    
    index_up = index_tip.y < index_pip.y
    middle_up = middle_tip.y < middle_pip.y
    
    if index_up and middle_up:
        x = int(index_tip.x * frame_w)
        y = int(index_tip.y * frame_h)
        return True, (x, y)
    return False, None

def predict_digit(canvas):
    """preprocess canvas and predict digit using mnist model."""
    canvas_copy = canvas.copy()
    
    non_white_pixels = np.sum(canvas_copy < 250)
    if non_white_pixels < 10:
        return 0, 0.0, np.zeros((28, 28), dtype=np.uint8)
    
    preprocessed_debug = preprocess_for_mnist(canvas_copy)
    bright_pixels = np.sum(preprocessed_debug > 50)
    if bright_pixels < 5:
        return 0, 0.0, preprocessed_debug
    
    model_input = preprocessed_debug.astype(np.float32) / 255.0
    model_input = np.clip(model_input, 0.0, 1.0)
    
    digit, probs, acts = predict_digit_from_28x28(model_input)
    confidence = float(probs[digit])
    return digit, confidence, preprocessed_debug


def save_trace_image(canvas, has_active_stroke, prediction=None):
    """save original and preprocessed trace images, and predict digit."""
    if not has_active_stroke:
        return prediction
    
    non_white = np.sum(canvas < 250)
    if non_white < 100:
        return prediction
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    try:
        digit, confidence, preprocessed = predict_digit(canvas)
        prediction = (digit, confidence)
    except Exception as e:
        print(f"prediction error: {e}")
        return prediction
    
    filename = os.path.join(TRACES_DIR, f"trace_{ts}.png")
    cv2.imwrite(filename, canvas)
    
    preprocessed_filename = os.path.join(TRACES_PREPROCESSED_DIR, f"trace_{ts}_preprocessed.png")
    cv2.imwrite(preprocessed_filename, preprocessed)
    
    return prediction

def create_preview(frame, canvas, fingertip_point, results, frame_w, frame_h, prediction=None):
    """create preview with camera feed, canvas overlay, fingertip indicator, and prediction."""
    preview = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    
    if fingertip_point is not None:
        cv2.circle(preview, fingertip_point, 8, (0, 255, 0), -1)
    elif results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        index_tip = hand_landmarks.landmark[8]
        x = int(index_tip.x * frame_w)
        y = int(index_tip.y * frame_h)
        cv2.circle(preview, (x, y), 8, (0, 0, 255), -1)
    
    if prediction is not None:
        digit, confidence = prediction
        text = f"prediction: {digit} ({confidence:.1%})"
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(preview, (10, 10), (20 + text_width, 40 + text_height), (0, 0, 0), -1)
        if confidence > 0.7:
            color = (0, 255, 0)
        elif confidence > 0.4:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)
        cv2.putText(preview, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    
    return preview


def main():
    """main application loop."""
    os.makedirs(TRACES_DIR, exist_ok=True)
    os.makedirs(TRACES_PREPROCESSED_DIR, exist_ok=True)
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    
    cap = open_camera()
    
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("could not read from camera")
    
    frame_h, frame_w = frame.shape[:2]
    canvas = np.ones((frame_h, frame_w, 3), dtype=np.uint8) * 255
    
    last_point = None
    smoothed_point = None
    last_time = None
    current_thickness = None
    no_hand_frames = 0
    has_active_stroke = False
    current_prediction = None
    
    print("running. press 'q' to quit, 'c' to clear canvas, 'p' to predict digit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            fingertip_point = None
            two_fingers_up = False
            
            if results.multi_hand_landmarks:
                two_fingers_up, fingertip_point = detect_two_fingers_up(
                    results.multi_hand_landmarks[0], frame_w, frame_h
                )
            
            if two_fingers_up and fingertip_point is not None:
                if not has_active_stroke:
                    current_prediction = None

                no_hand_frames = 0

                # 1. Smooth fingertip
                smoothed_point = smooth_point(smoothed_point, fingertip_point)

                # 2. Only draw if we have a previous point
                if last_point is not None and smoothed_point is not None:
                    dx = smoothed_point[0] - last_point[0]
                    dy = smoothed_point[1] - last_point[1]
                    dist_px = math.hypot(dx, dy)

                    # Time delta
                    now = time.time()
                    dt = now - last_time if last_time else 0.033
                    last_time = now
                    if dt <= 0: dt = 0.033

                    speed = dist_px / dt

                    # 3. Compute **target** thickness
                    if speed <= SPEED_FOR_MAX_THICK:
                        target_thickness = MAX_LINE_THICKNESS
                    elif speed >= SPEED_FOR_MIN_THICK:
                        target_thickness = MIN_LINE_THICKNESS
                    else:
                        t = (speed - SPEED_FOR_MAX_THICK) / (SPEED_FOR_MIN_THICK - SPEED_FOR_MAX_THICK)
                        target_thickness = int(
                            MAX_LINE_THICKNESS + t * (MIN_LINE_THICKNESS - MAX_LINE_THICKNESS)
                        )

                    # 4. **SMOOTH** thickness over time
                    if current_thickness is None:
                        current_thickness = target_thickness
                    else:
                        current_thickness = int(
                            THICKNESS_SMOOTHING_ALPHA * target_thickness +
                            (1 - THICKNESS_SMOOTHING_ALPHA) * current_thickness
                        )
                    thickness = max(MIN_LINE_THICKNESS, min(MAX_LINE_THICKNESS, current_thickness))

                    # 5. Jitter / jump control
                    if dist_px < MIN_MOVE_PIXELS:
                        pass
                    elif dist_px > MAX_JUMP_PIXELS:
                        last_point = smoothed_point
                    else:
                        # 6. DRAW with **smoothed** thickness
                        cv2.line(canvas, last_point, smoothed_point, LINE_COLOR, thickness, cv2.LINE_AA)
                        cv2.circle(canvas, smoothed_point, thickness // 2, LINE_COLOR, -1, cv2.LINE_AA)

                        has_active_stroke = True
                        last_point = smoothed_point
                else:
                    # First point
                    last_point = smoothed_point
                    last_time = time.time()
                    current_thickness = MAX_LINE_THICKNESS  # start thick

            else:
                last_point = None
                smoothed_point = None
                last_time = None
                current_thickness = None
                no_hand_frames += 1

                if no_hand_frames >= NO_HAND_FRAMES_THRESHOLD and has_active_stroke:
                    current_prediction = save_trace_image(canvas, has_active_stroke, current_prediction)
                    canvas.fill(255)
                    has_active_stroke = False
                    no_hand_frames = 0
            
            preview = create_preview(frame, canvas, fingertip_point, results, frame_w, frame_h, current_prediction)
            cv2.imshow("finger trace (preview)", preview)
            
            if cv2.getWindowProperty("finger trace (preview)", cv2.WND_PROP_VISIBLE) < 1:
                if has_active_stroke:
                    current_prediction = save_trace_image(canvas, has_active_stroke, current_prediction)
                break
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                if has_active_stroke:
                    current_prediction = save_trace_image(canvas, has_active_stroke, current_prediction)
                break
            elif key == ord('c'):
                canvas[:] = 255
                has_active_stroke = False
                last_point = None
                no_hand_frames = 0
                current_prediction = None
            elif key == ord('p'):
                if has_active_stroke:
                    try:
                        digit, confidence, preprocessed = predict_digit(canvas)
                        current_prediction = (digit, confidence)
                    except Exception as e:
                        print(f"prediction error: {e}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()
