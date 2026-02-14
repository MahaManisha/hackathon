
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import time
import os
import urllib.request
import queue
import sounddevice as sd
from deepface import DeepFace
from ultralytics import YOLO
import logging

# Suppress Logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# -------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# -------------------------------------------------------------------

EAR_THRESHOLD = 0.20
EAR_CONSEC_FRAMES = 30
HEAD_YAW_THRESHOLD = 20
HEAD_POSE_TIME_THRESHOLD = 2.0
EMOTION_INTERVAL = 10  # Process emotion every N frames

# Audio
SAMPLE_RATE = 16000
AUDIO_THRESHOLD = 0.02  # Low threshold
AUDIO_DURATION = 3.0    # 3 seconds of talking triggers alert
AUDIO_SILENCE_TOLERANCE = 1.0 # 1 sec pause allowed

# Risk Points
RISK_DROWSY = 5
RISK_DISTRACTION = 4
RISK_PHONE = 7
RISK_TALKING = 3
RISK_EMOTION_FATIGUE = 4
RISK_HIGH_FATIGUE = 6
RISK_PASSENGER = 3

# MediaPipe Setup
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MP_LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
], dtype=np.float64)

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = "face_landmarker.task"

# -------------------------------------------------------------------
# VOICE ALERT SYSTEM (State-Aware)
# -------------------------------------------------------------------
class VoiceAlertSystem:
    def __init__(self):
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        try:
            engine = pyttsx3.init()
            rate = engine.getProperty('rate')
            engine.setProperty('rate', rate - 20)
        except Exception as e:
            print(f"TTS Init Error: {e}")
            return
            
        while not self.stop_event.is_set():
            try:
                msg = self.queue.get(timeout=0.5)
                try:
                    engine.say(msg)
                    engine.runAndWait()
                except Exception as e:
                    print(f"TTS Say Error: {e}")
                finally:
                    self.queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                pass

    def say(self, message):
        if self.queue.qsize() < 2:
            self.queue.put(message)

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

# -------------------------------------------------------------------
# AUDIO MONITORING (Robust)
# -------------------------------------------------------------------
class AudioMonitor:
    def __init__(self):
        self.stop_event = threading.Event()
        self.talking_start_time = None
        self.last_talking_time = 0
        self.total_talking_duration = 0
        self.is_talking_excessively = False
        self.current_volume = 0.0 
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        def callback(indata, frames, time_info, status):
            if status: pass 
            
            volume_norm = np.linalg.norm(indata) * 10
            self.current_volume = volume_norm
            current_time = time.time()
            
            if volume_norm > AUDIO_THRESHOLD:
                if self.talking_start_time is None:
                    self.talking_start_time = current_time
                    self.total_talking_duration = 0
                else:
                    self.total_talking_duration = current_time - self.talking_start_time
                self.last_talking_time = current_time
            else:
                if self.talking_start_time is not None:
                    if (current_time - self.last_talking_time) > AUDIO_SILENCE_TOLERANCE:
                        self.talking_start_time = None
                        self.total_talking_duration = 0
                        self.is_talking_excessively = False

            if self.total_talking_duration > AUDIO_DURATION:
                self.is_talking_excessively = True
            else:
                self.is_talking_excessively = False

        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
                while not self.stop_event.is_set():
                    sd.sleep(100)
        except Exception as e:
            print(f"Mic Error: {e}")
            self.is_talking_excessively = False

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

# -------------------------------------------------------------------
# UTILITIES
# -------------------------------------------------------------------
def calculate_ear(landmarks, indices, img_w, img_h):
    coords = []
    for idx in indices:
        lm = landmarks[idx]
        coords.append((int(lm.x * img_w), int(lm.y * img_h)))
    p1, p2, p3, p4, p5, p6 = coords
    v1 = np.linalg.norm(np.array(p2)-np.array(p6))
    v2 = np.linalg.norm(np.array(p3)-np.array(p5))
    h_dist = np.linalg.norm(np.array(p1)-np.array(p4))
    if h_dist == 0: return 0.0, coords
    return (v1 + v2) / (2.0 * h_dist), coords

def get_head_pose(landmarks, w, h, cam_matrix, dist_coeffs):
    image_points = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in MP_LANDMARK_INDICES], dtype=np.float64)
    success, rot_vec, trans_vec = cv2.solvePnP(MODEL_POINTS_3D, image_points, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success: return 0, 0, 0
    rmat, _ = cv2.Rodrigues(rot_vec)
    euler = cv2.decomposeProjectionMatrix(np.hstack((rmat, trans_vec)))[6]
    return [x[0] for x in euler]

def draw_text(img, text, pos, color=(0,255,0), scale=0.6):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

# -------------------------------------------------------------------
# MAIN SYSTEM
# -------------------------------------------------------------------
class DriverMonitoringSystem:
    def __init__(self):
        # MediaPipe
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        if not os.path.exists(MODEL_PATH):
            try: urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            except: pass

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

        # Models
        print("Loading YOLO...")
        try: self.yolo = YOLO("yolov8n.pt")
        except: self.yolo = None
        
        # DeepFace Pre-initialization
        print("Initializing DeepFace...")
        try:
             dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
             DeepFace.analyze(dummy_img, actions=['emotion'], 
                            enforce_detection=False, 
                            detector_backend='skip')
             self.deepface_active = True
        except Exception:
             self.deepface_active = False

        # Subsystems
        self.voice = VoiceAlertSystem()
        self.audio = AudioMonitor()

        # --- NEW ALERT STATE ENGINE ---
        self.last_spoken_time = {} # Tracks cooldown per alert type
        self.alert_cooldown = 5.0 # Seconds between SAME alert
        self.current_display_alert = "" # For HUD text
        self.last_display_time = 0

        # Data
        self.ear_frame_counter = 0
        self.distraction_start = None
        self.prev_time = time.time()
        self.frame_count = 0
        
        # Flags
        self.drowsy_flag = False
        self.is_distracted = False
        self.phone_detected = False
        self.talking_flag = False
        self.current_emotion = "neutral"
        self.fatigue_flag = False
        self.passenger_detected = False

    def trigger_alert(self, alert_type, message, priority_level):
        """
        Smart Alert Logic:
        1. Checks per-alert-type cooldown.
        2. Allows high priority interrupts.
        """
        current_time = time.time()
        
        # Initialize cooldown tracker if new type
        if alert_type not in self.last_spoken_time:
            self.last_spoken_time[alert_type] = 0
        
        # Check Cooldown
        if current_time - self.last_spoken_time[alert_type] > self.alert_cooldown:
            # Speak
            self.voice.say(message)
            # Update timers
            self.last_spoken_time[alert_type] = current_time
            self.current_display_alert = message
            self.last_display_time = current_time
            return True
        
        return False

    def process_logic_and_alerts(self, risk_score):
        # PRIORITY ORDER: 
        # 1. High Risk (Generic or Specific)
        # 2. Fatigue (New Fusion)
        # 3. Phone
        # 4. Passenger (New)
        # 5. Drowsy
        # 6. Looking Away (Distraction)
        # 7. Talking
        # 8. Emotion
        
        spoken = False

        # 1. High Risk
        if risk_score >= 11:
            spoken = self.trigger_alert("HIGH_RISK", "High risk detected. Drive carefully.", 1)
        
        # 2. Fatigue
        if not spoken and self.fatigue_flag:
            spoken = self.trigger_alert("FATIGUE", "You appear tired. Please take a break.", 2)
            
        # 3. Phone
        if not spoken and self.phone_detected:
            spoken = self.trigger_alert("PHONE", "Phone usage detected. Please focus on driving.", 3)
            
        # 4. Passenger
        if not spoken and self.passenger_detected:
            spoken = self.trigger_alert("PASSENGER", "Passengers, please do not disturb the driver.", 4)
            
        # 5. Drowsy (Fallback)
        if not spoken and self.drowsy_flag:
            spoken = self.trigger_alert("DROWSY", "You seem sleepy. Please stay alert.", 5)
            
        # 6. Looking Away
        if not spoken and self.is_distracted:
            spoken = self.trigger_alert("LOOKING_AWAY", "Please keep your eyes on the road.", 6)
            
        # 7. Talking
        if not spoken and self.talking_flag:
            spoken = self.trigger_alert("TALKING", "Please reduce conversation and focus on driving.", 7)
            
        # 8. Emotion (Tired)
        if not spoken and self.current_emotion in ['sad', 'fear']:
            spoken = self.trigger_alert("EMOTION", "You look tired. Consider taking a short break.", 8)

    def run(self):
        cap = cv2.VideoCapture(0)
        print("System Active. Press 'q' to exit.")
        cam_matrix = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: continue
            
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            self.frame_count += 1
            current_time_ms = int(time.time() * 1000)
            
            if cam_matrix is None:
                cam_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype="double")
            
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = self.landmarker.detect_for_video(mp_img, current_time_ms)

            current_ear = 0.0
            yaw = 0.0
            face_coords = None
            
            if result.face_landmarks:
                fl = result.face_landmarks[0]
                
                # Coords for DeepFace
                xvals = [lm.x for lm in fl]
                yvals = [lm.y for lm in fl]
                face_coords = (max(0, int(min(xvals)*w)-20), max(0, int(min(yvals)*h)-20), 
                               min(w, int(max(xvals)*w)+20), min(h, int(max(yvals)*h)+20))

                # EAR
                l_ear, lc = calculate_ear(fl, LEFT_EYE_INDICES, w, h)
                r_ear, rc = calculate_ear(fl, RIGHT_EYE_INDICES, w, h)
                current_ear = (l_ear + r_ear) / 2.0
                
                if current_ear < EAR_THRESHOLD:
                    self.ear_frame_counter += 1
                    if self.ear_frame_counter >= EAR_CONSEC_FRAMES:
                        self.drowsy_flag = True
                else:
                    self.ear_frame_counter = 0
                    self.drowsy_flag = False

                # Head Pose
                _, yaw, _ = get_head_pose(fl, w, h, cam_matrix, np.zeros((4,1)))
                if abs(yaw) > HEAD_YAW_THRESHOLD:
                    if self.distraction_start is None: self.distraction_start = time.time()
                    elif time.time() - self.distraction_start > HEAD_POSE_TIME_THRESHOLD:
                        self.is_distracted = True
                else:
                    self.distraction_start = None
                    self.is_distracted = False
                    
                # Visualize
                for pt in lc+rc: cv2.circle(frame, pt, 1, (0,255,255), -1)

            # --- DEEPFACE ---
            if self.frame_count % EMOTION_INTERVAL == 0 and self.deepface_active and face_coords:
                try:
                    x1, y1, x2, y2 = face_coords
                    if x2>x1 and y2>y1:
                        crop = frame[y1:y2, x1:x2]
                        res = DeepFace.analyze(crop, actions=['emotion'], 
                                             detector_backend='skip', enforce_detection=False, silent=True)
                        if res: self.current_emotion = res[0]['dominant_emotion']
                except: pass

            # --- YOLO PHONE & PERSON ---
            self.phone_detected = False
            self.passenger_detected = False
            person_count = 0

            if self.yolo:
                try:
                    # Detect Person (0) and Cell Phone (67)
                    results = self.yolo(frame, verbose=False, classes=[0, 67], conf=0.4)
                    for r in results:
                        for box in r.boxes:
                            cls = int(box.cls[0])
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            if cls == 0: # Person
                                person_count += 1
                                # Draw person box slightly different if needed, or just let it trigger
                                # cv2.rectangle(frame, (x1, y1), (x2, y2), (255,100,0), 1)
                            
                            elif cls == 67: # Phone
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                                cv2.putText(frame, "PHONE", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                                self.phone_detected = True
                    
                    if person_count > 1:
                        self.passenger_detected = True

                except: pass

            # --- TALKING ---
            self.talking_flag = self.audio.is_talking_excessively

            # --- FATIGUE FUSION ---
            self.fatigue_flag = False
            if self.drowsy_flag or (self.current_emotion in ['sad', 'fear']):
                self.fatigue_flag = True

            # --- SCORING ---
            risk_score = 0
            if self.drowsy_flag: risk_score += RISK_DROWSY
            if self.is_distracted: risk_score += RISK_DISTRACTION
            if self.phone_detected: risk_score += RISK_PHONE
            if self.talking_flag: risk_score += RISK_TALKING
            if self.current_emotion in ['sad', 'fear']: risk_score += RISK_EMOTION_FATIGUE
            if self.passenger_detected: risk_score += RISK_PASSENGER
            
            # --- PROCESS ALERTS ---
            self.process_logic_and_alerts(risk_score)

            # --- HUD ---
            if risk_score >= 11: lev="HIGH RISK"; col=(0,0,255)
            elif risk_score >= 6: lev="WARNING"; col=(0,165,255)
            else: lev="SAFE"; col=(0,255,0)

            cv2.rectangle(frame, (0,0), (320, 280), (0,0,0), -1)
            y=30
            fps=1/(time.time()-self.prev_time); self.prev_time=time.time()
            
            draw_text(frame, f"FPS: {int(fps)}", (10,y), (255,255,255))
            y+=30; draw_text(frame, f"EAR: {current_ear:.2f}", (10,y))
            y+=30; draw_text(frame, f"Yaw: {int(yaw)}", (10,y))
            y+=30; draw_text(frame, f"Emotion: {self.current_emotion}", (10,y), (255,255,0))
            y+=30; draw_text(frame, f"Talking: {'YES' if self.talking_flag else 'NO'}", (10,y), (0,0,255) if self.talking_flag else (0,255,0))
            
            # Audio Bar
            bar_w = int(min(self.audio.current_volume * 1000, 100))
            cv2.rectangle(frame, (230, y-15), (230+bar_w, y-5), (0,255,255), -1)
            cv2.rectangle(frame, (230, y-15), (330, y-5), (255,255,255), 1)
            
            y+=30; draw_text(frame, f"Phone: {'YES' if self.phone_detected else 'NO'}", (10,y), (0,0,255) if self.phone_detected else (0,255,0))
            y+=30; draw_text(frame, f"Passenger: {'YES' if self.passenger_detected else 'NO'}", (10,y), (0,0,255) if self.passenger_detected else (0,255,0))
            
            if self.passenger_detected:
                cv2.putText(frame, "PASSENGER DETECTED", (320, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,140,255), 2)
            y+=30; draw_text(frame, f"Score: {risk_score}", (10,y), col)
            y+=30; draw_text(frame, f"Level: {lev}", (10,y), col)

            # Display active alert toast
            if self.current_display_alert and (time.time() - self.last_display_time < 3.0):
                tw, th = cv2.getTextSize(self.current_display_alert, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(frame, (w//2 - tw//2 - 10, h-50 - th - 10), (w//2 + tw//2 + 10, h-50 + 10), (0,0,0), -1)
                cv2.putText(frame, self.current_display_alert, (w//2 - tw//2, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            cv2.imshow("Driver Monitor (V3)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()
        self.voice.stop()
        self.audio.stop()

if __name__ == "__main__":
    DriverMonitoringSystem().run()
