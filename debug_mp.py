import mediapipe as mp
print(f"MediaPipe path: {mp.__file__}")
print(f"Dir(mp): {dir(mp)}")
try:
    print(f"mp.solutions: {mp.solutions}")
except AttributeError as e:
    print(f"Error accessing solutions: {e}")
