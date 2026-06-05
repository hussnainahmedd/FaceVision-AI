# FaceVision AI — Real-Time Face Recognition System

A real-time face recognition web application built with Python, OpenCV, and Flask. Detects faces via webcam, identifies registered individuals, and estimates age/gender using deep learning models.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.x-green?logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

- **Real-time face detection** using OpenCV Haar cascades (fast, CPU-friendly)
- **Face recognition** using `face_recognition` library (dlib, 128-d encodings)
- **Age estimation** using Caffe DNN models (8 age brackets)
- **Gender detection** using Caffe DNN models
- **Live webcam streaming** via MJPEG to the browser
- **Face registration** — upload a photo or capture directly from webcam
- **Premium dark UI** — glassmorphism design with live stats panel
- **REST API** for all operations

## Screenshots

> Run the app and open `http://localhost:5000` to see the UI.

## Tech Stack

| Component | Technology |
|---|---|
| Backend | Python, Flask |
| Face Detection | OpenCV Haar Cascades |
| Face Recognition | `face_recognition` (dlib) |
| Age/Gender | OpenCV DNN (Caffe models) |
| Frontend | HTML5, CSS3, Vanilla JS |
| Streaming | MJPEG over HTTP |

## Installation

### Prerequisites

- Python 3.10 or higher
- Webcam
- (Windows) Visual Studio Build Tools with "Desktop development with C++" workload (required for `dlib`)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/FaceVision-AI.git
cd FaceVision-AI

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download AI models
python download_models.py

# 5. Run the application
python app.py
```

Then open **http://localhost:5000** in your browser.

## Project Structure

```
FaceVision-AI/
├── app.py                  # Flask web server & REST API
├── camera.py               # Webcam capture & face processing
├── face_manager.py         # Known faces database (pickle)
├── download_models.py      # Script to download age/gender models
├── requirements.txt        # Python dependencies
├── .gitignore
├── models/                 # AI model files (downloaded separately)
│   ├── age_deploy.prototxt
│   ├── age_net.caffemodel
│   ├── gender_deploy.prototxt
│   └── gender_net.caffemodel
├── static/
│   ├── css/
│   │   └── style.css       # Premium dark glassmorphism theme
│   └── js/
│       └── main.js         # Frontend interactivity
├── templates/
│   └── index.html          # Main web UI
├── known_faces/            # Registered face images (created at runtime)
└── uploads/                # Temporary upload storage (created at runtime)
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/video_feed` | MJPEG video stream |
| `POST` | `/register_face` | Register face (multipart form: `name` + `image`) |
| `POST` | `/register_webcam` | Register face from webcam (JSON: `{"name": "..."}`) |
| `GET` | `/api/faces` | List all registered faces |
| `DELETE` | `/api/faces/<name>` | Remove a registered face |
| `GET` | `/api/faces/<name>/thumbnail` | Get face thumbnail image |
| `GET` | `/api/stats` | Live detection statistics |

## How It Works

1. **Capture** — A background thread continuously reads frames from the webcam.
2. **Detect** — OpenCV's Haar cascade detects face regions (fast, ~60 FPS).
3. **Encode** — `face_recognition` generates 128-dimensional face encodings.
4. **Match** — Encodings are compared against the registered faces database.
5. **Analyze** — OpenCV DNN models estimate age bracket and gender.
6. **Stream** — Annotated frames are streamed to the browser via MJPEG.

## Performance

The system uses a 3-thread architecture for smooth performance:

- **Thread 1 (Capture)**: Reads webcam frames continuously
- **Thread 2 (Processor)**: Runs face detection, recognition, and age/gender analysis
- **Main (Stream)**: Draws cached results + encodes JPEG (instant)

## License

MIT License — feel free to use, modify, and distribute.

## Acknowledgements

- [face_recognition](https://github.com/ageitgey/face_recognition) by Adam Geitgey
- [OpenCV](https://opencv.org/)
- Age/Gender models by [Gil Levi & Tal Hassner](https://github.com/GilLevi/AgeGenderDeepLearning)
