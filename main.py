import cv2
import mediapipe as mp
import time
import os
import urllib.request
from emotion_detector import EmotionDetector
from utils import calculate_ear, draw_landmarks, draw_text_info, draw_alert, VoiceAlert, LEFT_EYE_INDICES, RIGHT_EYE_INDICES

# Use MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = "face_landmarker.task"

# Drowsiness detection parameters
EAR_THRESHOLD = 0.18
CONSEC_FRAMES = 30

def download_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading model from {MODEL_URL}...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Model downloaded successfully.")
        except Exception as e:
            print(f"Error downloading model: {e}")
            return False
    return True

# Global variable to store latest detection results
latest_result = None

def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

def get_driver_state(emotion, is_drowsy):
    """
    Map emotion and drowsiness to driver state.
    """
    if is_drowsy:
        return "Tired (Drowsy!)"
    
    if emotion == 'sad':
        return "Tired"
    elif emotion == 'neutral':
        return "Alert (Neutral)"
    elif emotion == 'angry':
        return "Aggressive Risk"
    elif emotion == 'happy':
        return "Alert (Happy)"
    elif emotion == 'fear':
        return "Stressed"
    elif emotion == 'surprise':
        return "Distracted?"
    else:
        return f"Alert ({emotion})"

def main():
    if not download_model():
        print("Failed to ensure model file exists. Exiting.")
        return

    # Initialize VoiceAlert
    voice_alert = VoiceAlert()
    
    # Initialize Emotion Detector (OpenCV DNN + ONNX)
    emotion_detector = EmotionDetector()
    
    # Initialize FaceLandmarker with Live Stream mode
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=result_callback,
        num_faces=1
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        # Start Video Capture
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Starting video stream... Press 'q' to exit.")

        prev_time = 0
        
        # State variables for drowsiness detection
        counter = 0
        alarm_on = False
        voice_triggered = False
        
        # State variables for emotion detection
        # We don't need to detect emotion every single frame to save resources
        emotion_interval = 5 # Detect emotion every 5 frames
        frame_count = 0
        current_emotion = "neutral"
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally for a later selfie-view display
            image = cv2.flip(image, 1)
            frame_count += 1
            
            # --- Emotion Detection ---
            if frame_count % emotion_interval == 0:
                # FER library handles face splitting and grayscale conversion internally
                # top_emotion returns (emotion, score)
                try:
                    # Capture top emotion
                    emotion, score = emotion_detector.top_emotion(image)
                    if emotion:
                        current_emotion = emotion
                except Exception as e:
                    # In case of no face found or other errors
                    pass

            # Convert to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Detect asynchronously
            curr_time_ms = int(time.time() * 1000)
            landmarker.detect_async(mp_image, curr_time_ms)
            
            # Process latest result if available
            current_result = latest_result
            
            curr_ear = 0.0
            
            if current_result and current_result.face_landmarks:
                for face_landmarks in current_result.face_landmarks:
                    # Calculate EAR for both eyes
                    left_ear, left_coords = calculate_ear(face_landmarks, LEFT_EYE_INDICES, image.shape)
                    right_ear, right_coords = calculate_ear(face_landmarks, RIGHT_EYE_INDICES, image.shape)

                    # Average EAR
                    curr_ear = (left_ear + right_ear) / 2.0

                    # Draw eye landmarks
                    draw_landmarks(image, left_coords, color=(0, 255, 0), radius=2)
                    draw_landmarks(image, right_coords, color=(0, 255, 0), radius=2)
                    
                    # Check EAR threshold
                    if curr_ear < EAR_THRESHOLD:
                        counter += 1
                        
                        if counter >= CONSEC_FRAMES:
                            alarm_on = True
                            
                            if not voice_triggered:
                                voice_alert.speak("I think you feel sleepy. Please stay alert.")
                                voice_triggered = True
                    else:
                        counter = 0
                        alarm_on = False
                        voice_triggered = False

            # Draw Emotion and Driver State on Screen
            driver_state = get_driver_state(current_emotion, alarm_on)
            
            # Display Driver State
            cv2.putText(image, f"State: {driver_state}", (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
            # Display Info
            # Calculate FPS
            loop_curr_time = time.time()
            fps = 1 / (loop_curr_time - prev_time) if (loop_curr_time - prev_time) > 0 else 0
            prev_time = loop_curr_time
            
            draw_text_info(image, curr_ear, fps)
            
            # Draw drowsy warning if alarm is on
            if alarm_on:
                draw_alert(image)

            # Show the image
            cv2.imshow('Driver Drowsiness Detection', image)

            # Exit on 'q' key
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
