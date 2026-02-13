import mediapipe as mp
try:
    import mediapipe.python.solutions as solutions
    print("Found via mediapipe.python.solutions")
except ImportError:
    print("Not found via mediapipe.python.solutions")

try:
    from mediapipe import solutions
    print("Found via from mediapipe import solutions")
except ImportError:
    print("Not found via from mediapipe import solutions")
