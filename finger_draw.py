"""Finger drawing application with hand gesture recognition."""

import os
import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime

from config import (
    TRACES_DIR,
    TRACES_PREPROCESSED_DIR,
    LINE_COLOR,
    LINE_THICKNESS,
    NO_HAND_FRAMES_THRESHOLD,
    PREPROCESS_OUTPUT_SIZE,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MAX_NUM_HANDS,
    CAMERA_INDEX,
)
from preprocessing import preprocess_for_mnist
from mnist_model import predict_digit_from_28x28

def detect_two_fingers_up(hand_landmarks, frame_w, frame_h):
    """Check if index and middle fingers are extended."""
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
    """Preprocess canvas and predict digit using MNIST model."""
    # Make a copy to avoid modifying the original canvas
    canvas_copy = canvas.copy()
    
    # Check if canvas has any drawing (dark strokes on light background)
    # Canvas has white background (255) with black strokes (0)
    non_white_pixels = np.sum(canvas_copy < 250)
    if non_white_pixels < 10:  # Too few pixels, likely empty
        return 0, 0.0, np.zeros((28, 28), dtype=np.uint8)  # Dark background (MNIST format)
    
    # Output: black background (0) with white digit (255)
    preprocessed = preprocess_for_mnist(canvas_copy)
    
    bright_pixels = np.sum(preprocessed > 200)
    
    if bright_pixels < 10:
        return 0, 0.0, preprocessed
    
    preprocessed = np.clip(preprocessed, 0, 255)  # Ensure valid range
    
    preprocessed_normalized = preprocessed.astype(np.float32) / 255.0
    
    preprocessed_normalized = np.clip(preprocessed_normalized, 0.0, 1.0)
    
    digit, probs, acts = predict_digit_from_28x28(preprocessed_normalized)
    confidence = float(probs[digit])
    return digit, confidence, preprocessed


def save_trace_image(canvas, has_active_stroke, prediction=None):
    """Save original and preprocessed trace images, and predict digit."""
    if not has_active_stroke:
        return prediction
    
    non_white = np.sum(canvas < 250)
    if non_white < 100:
        return prediction  # Return old prediction, don't update
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    # Predict digit (this makes a copy internally)
    try:
        digit, confidence, preprocessed = predict_digit(canvas)
        prediction = (digit, confidence)
    except Exception as e:
        print(f"Prediction error: {e}")
        return prediction
    
    # Save original
    filename = os.path.join(TRACES_DIR, f"trace_{ts}.png")
    cv2.imwrite(filename, canvas)
    
    # Save preprocessed
    preprocessed_filename = os.path.join(TRACES_PREPROCESSED_DIR, f"trace_{ts}_preprocessed.png")
    cv2.imwrite(preprocessed_filename, preprocessed)
    
    return prediction


def create_preview(frame, canvas, fingertip_point, results, frame_w, frame_h, prediction=None):
    """Create preview with camera feed, canvas overlay, fingertip indicator, and prediction."""
    preview = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    
    if fingertip_point is not None:
        cv2.circle(preview, fingertip_point, 8, (0, 255, 0), -1)  # Green: drawing active
    elif results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        index_tip = hand_landmarks.landmark[8]
        x = int(index_tip.x * frame_w)
        y = int(index_tip.y * frame_h)
        cv2.circle(preview, (x, y), 8, (0, 0, 255), -1)  # Red: hand detected but not drawing
    
    # Display prediction if available
    if prediction is not None:
        digit, confidence = prediction
        text = f"Prediction: {digit} ({confidence:.1%})"
        # Draw background rectangle for text
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(preview, (10, 10), (20 + text_width, 40 + text_height), (0, 0, 0), -1)
        # Color based on confidence: green if high, yellow if medium, red if low
        if confidence > 0.7:
            color = (0, 255, 0)  # Green
        elif confidence > 0.4:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        cv2.putText(preview, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    
    return preview


def main():
    """Main application loop."""
    # Setup directories
    os.makedirs(TRACES_DIR, exist_ok=True)
    os.makedirs(TRACES_PREPROCESSED_DIR, exist_ok=True)
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    
    # Initialize camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera (index {CAMERA_INDEX})")
    
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read from camera")
    
    frame_h, frame_w = frame.shape[:2]
    canvas = np.ones((frame_h, frame_w, 3), dtype=np.uint8) * 255
    
    # Drawing state
    last_point = None
    no_hand_frames = 0
    has_active_stroke = False
    current_prediction = None
    
    print("Running. Press 'q' to quit, 'c' to clear canvas, 'p' to predict digit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # Mirror view
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            fingertip_point = None
            two_fingers_up = False
            
            # Detect gesture
            if results.multi_hand_landmarks:
                two_fingers_up, fingertip_point = detect_two_fingers_up(
                    results.multi_hand_landmarks[0], frame_w, frame_h
                )
            
            # Draw on canvas
            if two_fingers_up and fingertip_point is not None:
                # Clear prediction when starting a new drawing
                if not has_active_stroke:
                    current_prediction = None
                no_hand_frames = 0
                if last_point is not None:
                    cv2.line(canvas, last_point, fingertip_point, LINE_COLOR, LINE_THICKNESS)
                    has_active_stroke = True
                last_point = fingertip_point
            else:
                last_point = None
                no_hand_frames += 1
                
                # Auto-predict and save after hand is gone
                if no_hand_frames >= NO_HAND_FRAMES_THRESHOLD and has_active_stroke:
                    # Make prediction BEFORE clearing canvas
                    current_prediction = save_trace_image(canvas, has_active_stroke, current_prediction)
                    # Clear canvas AFTER prediction
                    canvas.fill(255)
                    has_active_stroke = False
                    no_hand_frames = 0
            
            # Display preview
            preview = create_preview(frame, canvas, fingertip_point, results, frame_w, frame_h, current_prediction)
            cv2.imshow("Finger Trace (preview)", preview)
            
            # Check window close
            if cv2.getWindowProperty("Finger Trace (preview)", cv2.WND_PROP_VISIBLE) < 1:
                if has_active_stroke:
                    save_trace_image(canvas, has_active_stroke, current_prediction)
                break
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                if has_active_stroke:
                    save_trace_image(canvas, has_active_stroke, current_prediction)
                break
            elif key == ord('c'):
                canvas[:] = 255
                has_active_stroke = False
                last_point = None
                no_hand_frames = 0
                current_prediction = None
            elif key == ord('p'):
                # Manual prediction trigger
                if has_active_stroke:
                    try:
                        digit, confidence, preprocessed = predict_digit(canvas)
                        current_prediction = (digit, confidence)
                    except Exception as e:
                        print(f"Prediction error: {e}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()
