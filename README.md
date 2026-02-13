# Driver Drowsiness Detection System

This project implements a real-time driver drowsiness detection system using OpenCV and MediaPipe.

## Features
- Real-time video capture from webcam
- Face Landmark detection using MediaPipe Tasks API (FaceLandmarker)
- Eye Aspect Ratio (EAR) calculation for blink/drowsiness monitoring
- Real-time visualization of eye landmarks and EAR values

## Requirements
- Python 3.x
- Webcam
- `face_landmarker.task` model file (automatically downloaded on first run)

## Installation

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script to start the detection system:

```bash
python main.py
```

- Press 'q' to quit the application.

## Project Structure
- `main.py`: Main application script handling video loop and processing.
- `utils.py`: Utility functions for geometry and drawing.
- `requirements.txt`: Python package dependencies.
