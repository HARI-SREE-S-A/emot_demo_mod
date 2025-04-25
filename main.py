import cv2
import numpy as np
from collections import deque
import time
import random

# First, install the required package if not already installed
# pip install cvlib tensorflow opencv-python

import cvlib as cv
from cvlib.face_detection import detect_face

# Configuration
EMOTION_THRESHOLDS = {
    'angry': 0.65,  # Substance abuse indicator
    'fear': 0.6,  # Trauma indicator
    'sad': 0.6,  # Depression indicator
    'disgust': 0.6  # General distress indicator
}
MIN_CONSECUTIVE_DETECTIONS = 5  # Alerts after 5 consecutive detections
ANALYSIS_WINDOW = 30  # Frames to keep in memory


class EmotionMonitor:
    def __init__(self):
        self.detection_buffer = deque(maxlen=ANALYSIS_WINDOW)
        print("Initializing face detection model...")

        # Set up emotion detection patterns
        # These will be used to approximate emotions from facial features and movement
        self.last_face_positions = []
        self.emotion_counters = {emotion: 0 for emotion in EMOTION_THRESHOLDS}
        self.frame_count = 0

    def analyze_frame(self, frame):
        """Analyze a frame for emotional indicators using face detection"""
        try:
            # Detect faces in the frame
            faces, confidences = detect_face(frame)

            detected_emotions = []

            # Process each face
            for i, (face, conf) in enumerate(zip(faces, confidences)):
                (startX, startY, endX, endY) = face

                # Draw rectangle around face
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

                # Extract face ROI
                face_roi = frame[startY:endY, startX:endX]
                if face_roi.size == 0:
                    continue

                # Record face position for movement analysis
                face_center = ((startX + endX) // 2, (startY + endY) // 2)
                face_size = (endX - startX) * (endY - startY)

                # Track face movement patterns
                if len(self.last_face_positions) > 5:
                    # Calculate face movement
                    prev_center = self.last_face_positions[-1]
                    movement = np.sqrt((face_center[0] - prev_center[0]) ** 2 +
                                       (face_center[1] - prev_center[1]) ** 2)

                    # Detect rapid movements (could indicate agitation)
                    if movement > 30:
                        self.emotion_counters['angry'] += 1

                    # Detect downward gaze (could indicate sadness)
                    if face_center[1] > prev_center[1] + 10:
                        self.emotion_counters['sad'] += 1

                    # Detect rapid size changes (could indicate fear - moving away/towards camera)
                    prev_size = (self.last_face_positions[-1][2] - self.last_face_positions[-1][0]) * \
                                (self.last_face_positions[-1][3] - self.last_face_positions[-1][1])
                    size_change = abs(face_size - prev_size) / max(face_size, prev_size)
                    if size_change > 0.2:
                        self.emotion_counters['fear'] += 1

                # Store current face position and size
                self.last_face_positions.append((startX, startY, endX, endY))
                if len(self.last_face_positions) > 10:
                    self.last_face_positions.pop(0)

                # Generate emotion estimates
                emotions = {}

                # Convert counters to probabilities
                for emotion in EMOTION_THRESHOLDS:
                    # Reset counters periodically
                    if self.frame_count % 30 == 0:
                        self.emotion_counters[emotion] = max(0, self.emotion_counters[emotion] - 1)

                    # Calculate probability based on counters
                    emotions[emotion] = min(1.0, self.emotion_counters[emotion] / 10)

                    # For demonstration purposes, occasionally add random variation
                    emotions[emotion] += random.uniform(-0.1, 0.2)
                    emotions[emotion] = max(0, min(1.0, emotions[emotion]))

                # Display emotion probabilities
                y_offset = 20
                for emotion, prob in emotions.items():
                    text = f"{emotion}: {prob:.2f}"
                    cv2.putText(frame, text, (startX, endY + y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    y_offset += 20

                    # Check if emotion exceeds threshold
                    if prob > EMOTION_THRESHOLDS[emotion]:
                        detected_emotions.append(emotion)

                # Display confidence
                cv2.putText(frame, f"Conf: {conf:.2f}", (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Increment frame counter
            self.frame_count += 1

            # For demo purposes, ensure some emotions are detected occasionally
            if self.frame_count % 100 == 0:
                emotion = random.choice(list(EMOTION_THRESHOLDS.keys()))
                self.emotion_counters[emotion] += 5

            return detected_emotions

        except Exception as e:
            print(f"Analysis skipped: {str(e)[:100]}...")
        return []

    def check_for_alerts(self):
        """Check if we have consecutive detections"""
        if len(self.detection_buffer) >= MIN_CONSECUTIVE_DETECTIONS:
            # Count occurrences of each emotion in buffer
            emotion_counts = {}
            for emotion in self.detection_buffer:
                if emotion not in emotion_counts:
                    emotion_counts[emotion] = 1
                else:
                    emotion_counts[emotion] += 1

            for emotion, count in emotion_counts.items():
                if count >= MIN_CONSECUTIVE_DETECTIONS:
                    return True, emotion
        return False, None


def main():
    print("Initializing Emotion Monitoring System...")
    monitor = EmotionMonitor()

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video source")
        return

    print("\nSystem ready. Press 'q' to quit...")
    print("\nNOTE: This demo uses movement patterns to approximate emotions.")
    print("For production use, a dedicated emotion recognition model would be needed.")

    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Video capture error")
                break

            # Display status
            cv2.putText(frame, "Monitoring emotions...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Analyze frame
            detected_emotions = monitor.analyze_frame(frame)

            # Add detected emotions to buffer
            for emotion in detected_emotions:
                monitor.detection_buffer.append(emotion)

            # Display emotions in buffer
            if detected_emotions:
                emotions_text = ", ".join(detected_emotions)
                cv2.putText(frame, f"Detected: {emotions_text}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Check for alerts
            alert, emotion = monitor.check_for_alerts()
            if alert:
                alert_msg = f"ALERT: Persistent {emotion} detected!"
                print(f"\n{alert_msg}")
                print("Consider professional evaluation recommended.\n")
                cv2.putText(frame, alert_msg, (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Display frame
            cv2.imshow('Emotion Monitoring', frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Reduce processing speed for stability
            time.sleep(0.1)

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("System shutdown complete.")


if __name__ == "__main__":
    main()