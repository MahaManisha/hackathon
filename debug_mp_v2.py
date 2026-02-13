import mediapipe as mp
print("Checking mp.tasks...")
try:
    print(f"mp.tasks: {mp.tasks}")
except AttributeError as e:
    print(f"Error accessing tasks: {e}")

print("Checking mp.solutions...")
try:
    print(f"mp.solutions: {mp.solutions}")
except AttributeError as e:
    print(f"Error accessing solutions: {e}")
