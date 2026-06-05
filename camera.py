"""
camera.py — Camera Capture & Face Processing (Fast + Age/Gender/Mood)

Uses OpenCV Haar cascade for instant face detection, face_recognition for
identification, and OpenCV DNN models for age/gender estimation.
Video streaming is completely decoupled and runs at full speed.
"""

import os
import threading
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
import face_recognition

from face_manager import FaceManager


# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------

_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

_AGE_BUCKETS = [
    "(0-2)", "(4-6)", "(8-12)", "(15-20)",
    "(25-32)", "(38-43)", "(48-53)", "(60-100)",
]
_GENDER_LIST = ["Male", "Female"]
_MODEL_MEAN = (78.4263377603, 87.7689143744, 114.895847746)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

_COLOR_KNOWN = (255, 212, 0)      # #00d4ff in BGR
_COLOR_UNKNOWN = (87, 71, 255)    # #ff4757 in BGR
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _draw_box(frame, top, right, bottom, left, name, conf, age, gender, color):
    """Draw bounding box with name and age/gender labels."""
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # Name label (above box)
    if name != "Unknown":
        label = f"{name} ({conf:.0%})"
    else:
        label = "Unknown"
    (tw, th), _ = cv2.getTextSize(label, _FONT, 0.5, 1)
    cv2.rectangle(frame, (left, top - th - 10), (left + tw + 6, top), color, cv2.FILLED)
    cv2.putText(frame, label, (left + 3, top - 5), _FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Age/Gender label (below box)
    if age or gender:
        info = ""
        if gender:
            info += gender
        if age:
            info += f", {age}" if info else age
        (tw2, th2), _ = cv2.getTextSize(info, _FONT, 0.45, 1)
        cv2.rectangle(frame, (left, bottom), (left + tw2 + 6, bottom + th2 + 10), (30, 30, 30), cv2.FILLED)
        cv2.putText(frame, info, (left + 3, bottom + th2 + 5), _FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Camera class (singleton)
# ---------------------------------------------------------------------------

class Camera:
    _instance = None
    _init_done = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, face_manager=None):
        if self._init_done:
            return
        Camera._init_done = True

        self._face_manager = face_manager or FaceManager()

        # Webcam
        self._cap = None
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._running = False

        # Haar cascade
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Age/Gender DNN models
        self._age_net = None
        self._gender_net = None
        self._load_models()

        # Results
        self._results_lock = threading.Lock()
        self._detected_faces: List[Dict[str, Any]] = []
        self._current_stats = {"face_count": 0, "faces": []}
        self._stats_lock = threading.Lock()

        self._start_capture()
        self._start_processor()

    # ------------------------------------------------------------------
    # Load DNN models
    # ------------------------------------------------------------------

    def _load_models(self):
        """Load age/gender Caffe models from the models/ directory."""
        age_proto = os.path.join(_MODELS_DIR, "age_deploy.prototxt")
        age_model = os.path.join(_MODELS_DIR, "age_net.caffemodel")
        gender_proto = os.path.join(_MODELS_DIR, "gender_deploy.prototxt")
        gender_model = os.path.join(_MODELS_DIR, "gender_net.caffemodel")

        try:
            if os.path.isfile(age_model) and os.path.isfile(age_proto):
                self._age_net = cv2.dnn.readNet(age_model, age_proto)
                print("[Camera] Age model loaded.")
            else:
                print(f"[Camera] Age model not found in {_MODELS_DIR}")

            if os.path.isfile(gender_model) and os.path.isfile(gender_proto):
                self._gender_net = cv2.dnn.readNet(gender_model, gender_proto)
                print("[Camera] Gender model loaded.")
            else:
                print(f"[Camera] Gender model not found in {_MODELS_DIR}")
        except Exception as exc:
            print(f"[Camera] Model load error (non-fatal): {exc}")

    # ------------------------------------------------------------------
    # Capture thread
    # ------------------------------------------------------------------

    def _start_capture(self):
        self._cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            print("[Camera] WARNING: Could not open webcam.")
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()
        print("[Camera] Capture started.")

    def _capture_loop(self):
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                time.sleep(0.1)
                continue
            ret, frame = self._cap.read()
            if ret:
                with self._frame_lock:
                    self._latest_frame = frame

    def get_raw_frame(self):
        """Return a copy of the latest raw frame."""
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    # ------------------------------------------------------------------
    # Processor thread
    # ------------------------------------------------------------------

    def _start_processor(self):
        threading.Thread(target=self._process_loop, daemon=True).start()
        print("[Camera] Processor started.")

    def _analyze_age_gender(self, frame, x, y, w, h):
        """Run age/gender DNN on a face crop. Returns (age_str, gender_str)."""
        if self._age_net is None or self._gender_net is None:
            return None, None

        try:
            fh, fw = frame.shape[:2]
            margin = 20
            y1 = max(0, y - margin)
            y2 = min(fh, y + h + margin)
            x1 = max(0, x - margin)
            x2 = min(fw, x + w + margin)
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                return None, None

            blob = cv2.dnn.blobFromImage(crop, 1.0, (227, 227), _MODEL_MEAN, swapRB=False)

            self._gender_net.setInput(blob)
            gender = _GENDER_LIST[self._gender_net.forward()[0].argmax()]

            self._age_net.setInput(blob)
            age = _AGE_BUCKETS[self._age_net.forward()[0].argmax()]

            return age, gender
        except Exception:
            return None, None

    def _process_loop(self):
        """Detect faces, identify them, estimate age/gender."""
        cycle = 0
        age_gender_cache: Dict[str, Tuple[Optional[str], Optional[str]]] = {}

        while self._running:
            with self._frame_lock:
                if self._latest_frame is None:
                    time.sleep(0.05)
                    continue
                frame = self._latest_frame.copy()

            cycle += 1

            # --- Haar cascade detection (fast) ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            haar_faces = self._face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5,
                minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE
            )

            if len(haar_faces) == 0:
                with self._results_lock:
                    self._detected_faces = []
                with self._stats_lock:
                    self._current_stats = {"face_count": 0, "faces": []}
                age_gender_cache.clear()
                time.sleep(0.1)
                continue

            # --- face_recognition encoding at half resolution ---
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            fr_locs = [(y // 2, (x + w) // 2, (y + h) // 2, x // 2) for (x, y, w, h) in haar_faces]

            try:
                encodings = face_recognition.face_encodings(rgb_small, fr_locs)
            except Exception:
                encodings = []

            # --- Build results ---
            faces_result = []
            for idx, (x, y, w, h) in enumerate(haar_faces):
                top, right, bottom, left = int(y), int(x + w), int(y + h), int(x)

                # Identity
                if idx < len(encodings):
                    name, conf = self._face_manager.find_match(encodings[idx])
                else:
                    name, conf = "Unknown", 0.0

                # Age/Gender — run every 15 cycles to save CPU
                cache_key = f"{top//50}_{left//50}"
                if cycle % 15 == 0 or cache_key not in age_gender_cache:
                    age, gender = self._analyze_age_gender(frame, x, y, w, h)
                    if age is not None:
                        age_gender_cache[cache_key] = (age, gender)
                else:
                    age, gender = age_gender_cache.get(cache_key, (None, None))

                faces_result.append({
                    "name": name,
                    "confidence": float(conf),
                    "age": age,
                    "gender": gender,
                    "box": (top, right, bottom, left),
                })

            # --- Store ---
            with self._results_lock:
                self._detected_faces = list(faces_result)

            stats_faces = [
                {
                    "name": f["name"],
                    "confidence": f["confidence"],
                    "age": f["age"],
                    "gender": f["gender"],
                    "box": [f["box"][0], f["box"][1], f["box"][2], f["box"][3]],
                }
                for f in faces_result
            ]
            with self._stats_lock:
                self._current_stats = {"face_count": len(stats_faces), "faces": stats_faces}

            time.sleep(0.05)

    # ------------------------------------------------------------------
    # Stream
    # ------------------------------------------------------------------

    def get_processed_frame(self):
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            frame = self._latest_frame.copy()

        with self._results_lock:
            faces = list(self._detected_faces)

        for f in faces:
            top, right, bottom, left = f["box"]
            color = _COLOR_KNOWN if f["name"] != "Unknown" else _COLOR_UNKNOWN
            _draw_box(frame, top, right, bottom, left,
                      f["name"], f["confidence"], f.get("age"), f.get("gender"), color)

        ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return jpeg.tobytes() if ret else None

    def generate_frames(self):
        while True:
            jpeg = self.get_processed_frame()
            if jpeg is None:
                time.sleep(0.03)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(0.033)

    @property
    def current_stats(self):
        with self._stats_lock:
            # Deep copy to avoid race conditions with JSON serialization
            import copy
            return copy.deepcopy(self._current_stats)

    def __del__(self):
        self._running = False
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
