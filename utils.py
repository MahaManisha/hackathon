
import math
import numpy as np
import cv2
import pyttsx3
import threading

# Define indices for Left and Right Eye landmarks based on MediaPipe Face Mesh
# Order: P1, P2, P3, P4, P5, P6
# P1, P4: Horizontal corners
# P2, P6: First vertical pair
# P3, P5: Second vertical pair
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

def calculate_euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.
    """
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def calculate_ear(landmarks, indices, image_shape):
    """
    Calculate the Eye Aspect Ratio (EAR) for a given eye.
    
    EAR = (|P2-P6| + |P3-P5|) / (2 * |P1-P4|)
    
    Args:
        landmarks: List of MediaPipe landmarks.
        indices: List of 6 standard eye landmark indices.
        image_shape: Tuple (height, width, channels) to denormalize coordinates.
        
    Returns:
        float: Calculated EAR value.
        list: List of (x, y) coordinates for the eye landmarks.
    """
    h, w, _ = image_shape
    coords = []
    
    # Extract coordinates for the specific eye indices
    for idx in indices:
        lm = landmarks[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        coords.append((x, y))
    
    # Unpack coordinates (P1..P6)
    p1, p2, p3, p4, p5, p6 = coords
    
    # Calculate vertical distances
    vertical_dist1 = calculate_euclidean_distance(p2, p6)
    vertical_dist2 = calculate_euclidean_distance(p3, p5)
    
    # Calculate horizontal distance
    horizontal_dist = calculate_euclidean_distance(p1, p4)
    
    if horizontal_dist == 0:
        return 0.0, coords

    # Calculate EAR
    ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
    
    return ear, coords

def draw_landmarks(image, landmarks_list, color=(0, 255, 0), radius=2):
    """
    Draw landmarks on the image.
    
    Args:
        image: The image to draw on.
        landmarks_list: List of (x, y) coordinates.
        color: BGR color tuple.
        radius: Radius of the drawn points.
    """
    for point in landmarks_list:
        cv2.circle(image, point, radius, color, -1)
        



class VoiceAlert:
    """
    Handles Text-to-Speech alerts in a separate thread.
    Initializes the engine inside the thread to avoid COM/threading issues on Windows.
    """
    def __init__(self):
        self._is_speaking = False

    def speak(self, text):
        if not self._is_speaking:
            self._is_speaking = True
            threading.Thread(target=self._speak_thread, args=(text,), daemon=True).start()

    def _speak_thread(self, text):
        try:
            # Initialize engine inside the thread
            engine = pyttsx3.init()
            
            # Optional: Slower rate for clearer alert
            rate = engine.getProperty('rate')
            engine.setProperty('rate', rate - 30)

            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
        finally:
            self._is_speaking = False

def draw_text_info(image, ear, fps):
    """
    Draw EAR and FPS info on the frame.
    """
    cv2.putText(image, f"EAR: {ear:.2f}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(image, f"FPS: {int(fps)}", (image.shape[1] - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def draw_alert(image):
    """
    Draw a large red warning on the screen.
    """
    h, w, _ = image.shape
    text = "DROWSINESS ALERT!"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.5
    thickness = 3
    color = (0, 0, 255) # Red

    # Center the text
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (h + text_size[1]) // 2

    cv2.putText(image, text, (text_x, text_y), font, scale, color, thickness)
