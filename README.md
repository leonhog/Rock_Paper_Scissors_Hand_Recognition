# Rock Paper Scissors Hand Recognition

Real-time Rock Paper Scissors recognition using a webcam, MediaPipe Hand Landmarker, and an XGBoost classification model.

## Overview

The application detects up to two hands, extracts hand landmarks, and classifies each gesture as Rock, Paper, or Scissors. When two hands are detected, the program determines the winner and displays the result using colored bounding boxes.

## Project Structure

```text
├── main.py
├── drawing_function.py
├── hand_function.py
├── model_function.py
├── hand_landmarker.task
├── xgboost_model.json
└── README.md
```

* **`main.py`** — Main application and camera processing.
* **`hand_function.py`** — Hand landmark processing and feature extraction.
* **`model_function.py`** — XGBoost model loading and gesture prediction.
* **`drawing_function.py`** — Visualization of landmarks, bounding boxes, and predictions.

## Requirements

Python 3.9+ and the following packages:

```bash
pip install opencv-python mediapipe numpy xgboost
```

The project also requires:

* `hand_landmarker.task` — MediaPipe hand detection model.
* `xgboost_model.json` — Trained XGBoost classification model.

## Usage

Run the application with:

```bash
python main.py
```

The webcam feed will open and display the detected gestures. Press `Q` to exit.

## Technologies

* Python
* OpenCV
* MediaPipe
* NumPy
* XGBoost


