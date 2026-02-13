
try:
    from emotion_detector import EmotionDetector
    import cv2
    import numpy as np
    
    detector = EmotionDetector()
    print("EmotionDetector class initialized successfully.")
    
    # Create a dummy image (gray 300x300)
    dummy_image = np.zeros((300, 300, 3), dtype=np.uint8)
    # Draw a face-like rectangle
    cv2.rectangle(dummy_image, (50, 50), (250, 250), (255, 255, 255), -1)
    
    emotion, score = detector.top_emotion(dummy_image)
    print(f"Test run result: Emotion={emotion}, Score={score}")
    
except Exception as e:
    print(f"Test failed: {e}")
