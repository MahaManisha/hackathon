
import cv2
import numpy as np
import os

class EmotionDetector:
    def __init__(self, model_path="emotion-ferplus-8.onnx"):
        # FERPlus emotional labels
        self.emotion_labels = ['neutral', 'happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'contempt']
        
        # Load Face Detection Cascade (OpenCV standard)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Load ONNX Model
        if os.path.exists(model_path):
            try:
                self.net = cv2.dnn.readNetFromONNX(model_path)
                print(f"Loaded emotion model from {model_path}")
            except Exception as e:
                print(f"Error loading emotion model: {e}")
                self.net = None
        else:
            print(f"Warning: Emotion model {model_path} not found. Emotion detection will be disabled.")
            self.net = None

    def top_emotion(self, image):
        """
        Detects face and predict top emotion.
        Returns: (emotion_label, score) or (None, 0.0) if failed.
        """
        if self.net is None:
            return None, 0.0
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Use scaleFactor 1.1, minNeighbors 5, minSize (30, 30)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None, 0.0
            
        # Process the largest face found
        # (x, y, w, h)
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        
        roi_gray = gray[y:y+h, x:x+w]
        
        try:
            # Resize to model input size 64x64
            roi_gray = cv2.resize(roi_gray, (64, 64))
            
            # Preprocess for FERPlus: Image should be 64x64, grayscale
            # Input blob: (1, 1, 64, 64)
            blob = cv2.dnn.blobFromImage(roi_gray, 1.0, (64, 64), (0, 0, 0), swapRB=False, crop=False)
            
            self.net.setInput(blob)
            scores = self.net.forward()
            
            # Helper for softmax
            def softmax(x):
                # Subtract max for numerical stability
                e_x = np.exp(x - np.max(x))
                return e_x / e_x.sum(axis=1)

            scores = softmax(scores)[0]
            
            max_index = np.argmax(scores)
            emotion = self.emotion_labels[max_index]
            score = scores[max_index]
            
            return emotion, score
            
        except Exception as e:
            # Handle potential resize errors or other issues
            return None, 0.0
