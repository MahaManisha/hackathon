
import cv2
import numpy as np

print(f"OpenCV version: {cv2.__version__}")
try:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Just creating a window to test if the function exists
    cv2.imshow('Test Window', img)
    cv2.waitKey(100) # Wait 100ms
    cv2.destroyAllWindows()
    print("SUCCESS: cv2.imshow() is working correctly.")
except cv2.error as e:
    print(f"FAILURE: OpenCV error: {e}")
except Exception as e:
    print(f"FAILURE: General error: {e}")
