
try:
    from deepface import DeepFace
    print("DeepFace imported successfully!")
except ImportError as e:
    print(f"Failed to import DeepFace: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
