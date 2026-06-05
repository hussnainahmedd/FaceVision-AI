"""
face_manager.py — Known Faces Database Manager

Provides a thread-safe manager for registering, removing, and matching
face encodings against a persistent pickle-based database.
"""

import os
import pickle
import shutil
import threading
from typing import List, Optional, Tuple, Union

import cv2
import face_recognition
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATABASE_PATH = "known_faces.pkl"
KNOWN_FACES_DIR = "known_faces"


class FaceManager:
    """Thread-safe manager for known face encodings.

    Data is persisted to a pickle file with the structure::

        {
            "names":     [str, ...],
            "encodings": [numpy_array_128d, ...],
        }

    Source images are stored in the ``known_faces/`` directory so they can be
    reviewed or re-encoded later.
    """

    def __init__(
        self,
        database_path: str = DATABASE_PATH,
        known_faces_dir: str = KNOWN_FACES_DIR,
    ) -> None:
        self._database_path = database_path
        self._known_faces_dir = known_faces_dir
        self._lock = threading.Lock()

        # Ensure the storage directory exists.
        os.makedirs(self._known_faces_dir, exist_ok=True)

        # Load existing database or start fresh.
        self._data: dict = self._load_database()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load_database(self) -> dict:
        """Load the face database from disk, or return an empty structure."""
        if os.path.exists(self._database_path):
            try:
                with open(self._database_path, "rb") as fh:
                    data = pickle.load(fh)
                # Basic sanity check
                if (
                    isinstance(data, dict)
                    and "names" in data
                    and "encodings" in data
                    and len(data["names"]) == len(data["encodings"])
                ):
                    print(
                        f"[FaceManager] Loaded {len(data['names'])} known face(s) "
                        f"from '{self._database_path}'."
                    )
                    return data
                else:
                    print(
                        "[FaceManager] Database file has unexpected structure — "
                        "starting fresh."
                    )
            except Exception as exc:
                print(f"[FaceManager] Failed to load database: {exc}")

        return {"names": [], "encodings": []}

    def _save_database(self) -> None:
        """Persist the current face database to disk.

        Must be called while the lock is held.
        """
        try:
            with open(self._database_path, "wb") as fh:
                pickle.dump(self._data, fh)
        except Exception as exc:
            print(f"[FaceManager] Failed to save database: {exc}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_face(
        self,
        name: str,
        image: Union[str, np.ndarray],
    ) -> Tuple[bool, str]:
        """Register a new face encoding in the database.

        Parameters
        ----------
        name:
            A human-readable label for this face.
        image:
            Either a file-system path to an image **or** a NumPy array
            (BGR or RGB) containing at least one face.

        Returns
        -------
        (success, message) tuple.
        """
        if not name or not name.strip():
            return False, "Name cannot be empty."

        name = name.strip()

        try:
            # --- Load / normalise the image ----------------------------------
            if isinstance(image, str):
                if not os.path.isfile(image):
                    return False, f"Image file not found: {image}"
                img_bgr = cv2.imread(image)
                if img_bgr is None:
                    return False, f"Failed to read image: {image}"
            elif isinstance(image, np.ndarray):
                img_bgr = image
            else:
                return False, "image must be a file path (str) or numpy array."

            # face_recognition expects RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # --- Extract encoding --------------------------------------------
            face_locations = face_recognition.face_locations(img_rgb, model="hog")
            if not face_locations:
                return False, "No face detected in the provided image."

            encodings = face_recognition.face_encodings(img_rgb, face_locations)
            if not encodings:
                return False, "Could not compute face encoding."

            # Use the first (largest / most prominent) face.
            encoding = encodings[0]

            # --- Save source image -------------------------------------------
            dest_path = os.path.join(self._known_faces_dir, f"{name}.jpg")
            cv2.imwrite(dest_path, img_bgr)

            # --- Update database (thread-safe) --------------------------------
            with self._lock:
                # If a face with this name already exists, replace it.
                if name in self._data["names"]:
                    idx = self._data["names"].index(name)
                    self._data["encodings"][idx] = encoding
                else:
                    self._data["names"].append(name)
                    self._data["encodings"].append(encoding)
                self._save_database()

            print(f"[FaceManager] Registered face: '{name}'")
            return True, f"Face '{name}' registered successfully."

        except Exception as exc:
            return False, f"Error registering face: {exc}"

    def remove_face(self, name: str) -> Tuple[bool, str]:
        """Remove a face from the database and delete its source image.

        Parameters
        ----------
        name:
            The label of the face to remove.

        Returns
        -------
        (success, message) tuple.
        """
        if not name or not name.strip():
            return False, "Name cannot be empty."

        name = name.strip()

        with self._lock:
            if name not in self._data["names"]:
                return False, f"No face registered with the name '{name}'."

            idx = self._data["names"].index(name)
            self._data["names"].pop(idx)
            self._data["encodings"].pop(idx)
            self._save_database()

        # Remove source image (best-effort).
        image_path = os.path.join(self._known_faces_dir, f"{name}.jpg")
        try:
            if os.path.isfile(image_path):
                os.remove(image_path)
        except OSError as exc:
            print(f"[FaceManager] Warning — could not delete image: {exc}")

        print(f"[FaceManager] Removed face: '{name}'")
        return True, f"Face '{name}' removed successfully."

    def get_all_faces(self) -> List[str]:
        """Return a sorted list of all registered face names."""
        with self._lock:
            return sorted(self._data["names"])

    def find_match(
        self,
        encoding: np.ndarray,
        tolerance: float = 0.6,
    ) -> Tuple[str, float]:
        """Find the best match for *encoding* among known faces.

        Parameters
        ----------
        encoding:
            A 128-d face encoding vector.
        tolerance:
            Maximum distance to consider a match (lower = stricter).

        Returns
        -------
        (name, confidence) where *confidence* is ``1 - distance``
        (clamped to [0, 1]).  Returns ``("Unknown", 0.0)`` if no match.
        """
        with self._lock:
            if not self._data["encodings"]:
                return ("Unknown", 0.0)

            known_encodings = self._data["encodings"]
            known_names = self._data["names"]

        # Compute distances and boolean matches.
        distances = face_recognition.face_distance(known_encodings, encoding)
        matches = face_recognition.compare_faces(
            known_encodings, encoding, tolerance=tolerance
        )

        best_idx: Optional[int] = None
        best_distance = float("inf")

        for idx, (is_match, dist) in enumerate(zip(matches, distances)):
            if is_match and dist < best_distance:
                best_distance = dist
                best_idx = idx

        if best_idx is not None:
            confidence = round(float(max(0.0, min(1.0, 1.0 - best_distance))), 4)
            return (known_names[best_idx], confidence)

        return ("Unknown", 0.0)
