"""
app.py — Flask Web Server for Real-Time Face Recognition

Exposes REST endpoints for video streaming (MJPEG), face registration /
removal, and live detection statistics.
"""

import os

import cv2

from flask import Flask, Response, jsonify, render_template, request, send_file

from camera import Camera
from face_manager import FaceManager

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

UPLOAD_DIR = "uploads"
KNOWN_FACES_DIR = "known_faces"

# ---------------------------------------------------------------------------
# Bootstrap directories
# ---------------------------------------------------------------------------

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Application globals
# ---------------------------------------------------------------------------

app = Flask(__name__)

face_manager = FaceManager(known_faces_dir=KNOWN_FACES_DIR)
camera = Camera(face_manager=face_manager)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    """Serve the main UI page."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """Stream annotated webcam frames as MJPEG.

    The client should use an ``<img>`` tag with its ``src`` pointing here,
    or open this URL directly in a browser.
    """
    return Response(
        camera.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/register_face", methods=["POST"])
def register_face():
    """Register a new face from an uploaded image.

    Expects ``multipart/form-data`` with:
    - ``name``  (text field)  — label for the face
    - ``image`` (file field)  — photo containing the face

    Returns JSON::

        {"success": bool, "message": str}
    """
    try:
        name = request.form.get("name", "").strip()
        if not name:
            return jsonify({"success": False, "message": "Name is required."}), 400

        if "image" not in request.files:
            return jsonify({"success": False, "message": "Image file is required."}), 400

        image_file = request.files["image"]
        if image_file.filename == "":
            return jsonify({"success": False, "message": "No file selected."}), 400

        # Save the uploaded file temporarily.
        safe_filename = "".join(
            c if c.isalnum() or c in ("_", "-", ".") else "_"
            for c in image_file.filename
        )
        save_path = os.path.join(UPLOAD_DIR, safe_filename)
        image_file.save(save_path)

        # Register via FaceManager.
        success, message = face_manager.register_face(name, save_path)

        # Clean up the temporary upload after registration.
        try:
            if os.path.isfile(save_path):
                os.remove(save_path)
        except OSError:
            pass

        status_code = 200 if success else 400
        return jsonify({"success": success, "message": message}), status_code

    except Exception as exc:
        return (
            jsonify({"success": False, "message": f"Server error: {exc}"}),
            500,
        )


@app.route("/api/faces", methods=["GET"])
def list_faces():
    """Return a JSON list of all registered face names.

    Response::

        {"success": true, "faces": ["Alice", "Bob"]}
    """
    try:
        faces = face_manager.get_all_faces()
        return jsonify({"success": True, "faces": faces}), 200
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 500


@app.route("/api/faces/<name>", methods=["DELETE"])
def delete_face(name: str):
    """Remove a registered face by name.

    Returns JSON::

        {"success": bool, "message": str}
    """
    try:
        success, message = face_manager.remove_face(name)
        status_code = 200 if success else 404
        return jsonify({"success": success, "message": message}), status_code
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 500


@app.route("/api/stats", methods=["GET"])
def stats():
    """Return live detection statistics (face count, names, ages, etc.).

    Response::

        {
            "face_count": 2,
            "faces": [
                {
                    "name": "Alice",
                    "confidence": 0.72,
                    "age": 28,
                    "gender": "Woman",
                    "box": [top, right, bottom, left]
                },
                ...
            ]
        }
    """
    try:
        return jsonify(camera.current_stats), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/faces/<name>/thumbnail", methods=["GET"])
def face_thumbnail(name: str):
    """Serve the stored image for a registered face.

    Returns the JPEG image file from the ``known_faces/`` directory.
    """
    image_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
    if not os.path.isfile(image_path):
        return jsonify({"success": False, "message": "Thumbnail not found."}), 404
    return send_file(image_path, mimetype="image/jpeg")


@app.route("/register_webcam", methods=["POST"])
def register_webcam():
    """Register a face by capturing a snapshot from the live webcam feed.

    Expects JSON: {"name": "Person Name"}

    This gives much better recognition accuracy since the face encoding
    is created under the same lighting/angle as the live feed.
    """
    try:
        data = request.get_json(silent=True) or {}
        name = data.get("name", "").strip()
        if not name:
            return jsonify({"success": False, "message": "Name is required."}), 400

        frame = camera.get_raw_frame()
        if frame is None:
            return jsonify({"success": False, "message": "Camera not ready."}), 500

        # Save the snapshot temporarily
        snap_path = os.path.join(UPLOAD_DIR, f"snap_{name}.jpg")
        cv2.imwrite(snap_path, frame)

        # Register via FaceManager
        success, message = face_manager.register_face(name, snap_path)

        # Cleanup temp file
        try:
            if os.path.isfile(snap_path):
                os.remove(snap_path)
        except OSError:
            pass

        status_code = 200 if success else 400
        return jsonify({"success": success, "message": message}), status_code

    except Exception as exc:
        return jsonify({"success": False, "message": f"Server error: {exc}"}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _print_banner() -> None:
    """Print a friendly startup banner."""
    banner = """
    ==================================================
           Face Recognition Web App
    --------------------------------------------------
      Server running at:
        -> http://localhost:5000
        -> http://0.0.0.0:5000

      Endpoints:
        GET  /             - Web UI
        GET  /video_feed   - MJPEG stream
        POST /register_face - Register a face
        GET  /api/faces    - List registered faces
        DEL  /api/faces/<n> - Remove a face
        GET  /api/stats    - Live detection stats
    ==================================================
    """
    print(banner)


if __name__ == "__main__":
    _print_banner()
    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True,
        debug=False,
    )
