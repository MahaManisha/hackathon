
import mediapipe as mp
try:
    print(f"File: {mp.__file__}")
    print(f"Solutions: {mp.solutions}")
    print("Success")
except AttributeError as e:
    print(f"Error: {e}")
    print(f"Dir: {dir(mp)}")
