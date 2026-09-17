import base64
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template, request

from model import SymbolCNN

BASE_DIR = __file__
IMAGE_SIZE = 64

with open("labels.txt", "r", encoding="utf-8") as label_file:
    symbols = [line.strip() for line in label_file if line.strip()]

model = SymbolCNN(num_classes=len(symbols))
model.load_state_dict(torch.load("Symbol_model.pth", map_location="cpu"))
model.eval()

app = Flask(__name__)
state_lock = threading.Lock()
stop_event = threading.Event()

camera = cv2.VideoCapture(0)
latest_frame = None
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
prev_point = None
prev_rotate_point = None
rotation_x = 0.0
rotation_y = 0.0
prediction = None
confidence = 0.0
mode = "draw"

components = {
    "resistor": "An electronic component that limits electrical current.",
    "gear": "A rotating part with teeth that transmits force and speed.",
    "pendulum": "A weight on a rod that swings freely under gravity.",
    "angle": "The space between two lines that meet at a point.",
}


def classify_canvas():
    image = cv2.resize(canvas, (IMAGE_SIZE, IMAGE_SIZE))
    image = image.astype(np.float32) / 255.0
    tensor = torch.tensor(image.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    index = int(torch.argmax(probabilities).item())
    return symbols[index], float(probabilities[index].item() * 100)


def camera_loop():
    global latest_frame, prev_point, prev_rotate_point, rotation_x, rotation_y, canvas, prediction, confidence
    while not stop_event.is_set():
        success, frame = camera.read()
        if not success:
            time.sleep(0.05)
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        point = None
        if results.multi_hand_landmarks:
            tip = results.multi_hand_landmarks[0].landmark[8]
            point = (int(tip.x * canvas.shape[1]), int(tip.y * canvas.shape[0]))
            if mode == "draw":
                if prev_point is not None:
                    cv2.line(canvas, prev_point, point, (65, 235, 220), 7, cv2.LINE_AA)
                prev_point = point
            else:
                if prev_rotate_point is not None:
                    rotation_y += (point[0] - prev_rotate_point[0]) * 0.012
                    rotation_x += (point[1] - prev_rotate_point[1]) * 0.012
                    rotation_x = max(-1.45, min(1.45, rotation_x))
                prev_rotate_point = point
                prev_point = None
        else:
            prev_point = None
            prev_rotate_point = None
        with state_lock:
            latest_frame = frame.copy()


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.65, min_tracking_confidence=0.6)
threading.Thread(target=camera_loop, daemon=True).start()


def jpeg_bytes(image):
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return encoded.tobytes() if success else b""


def current_state():
    with state_lock:
        frame = latest_frame.copy() if latest_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        trace = canvas.copy()
        current_prediction = prediction
        current_confidence = confidence
        current_rotation = (rotation_x, rotation_y)
    return {
        "prediction": current_prediction,
        "confidence": round(current_confidence, 1),
        "description": components.get(current_prediction, "Draw a gesture, then classify it."),
        "mode": mode,
        "rotation": {"x": round(current_rotation[0], 4), "y": round(current_rotation[1], 4)},
        "trace": base64.b64encode(jpeg_bytes(trace)).decode("ascii"),
        "camera_ready": latest_frame is not None,
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/state")
def api_state():
    return jsonify(current_state())


@app.post("/api/classify")
def api_classify():
    global prediction, confidence, mode, prev_rotate_point, rotation_x, rotation_y
    with state_lock:
        prediction, confidence = classify_canvas()
        mode = "rotate"
        prev_rotate_point = None
        rotation_x = 0.0
        rotation_y = 0.0
    return jsonify(current_state())


@app.post("/api/clear")
def api_clear():
    global canvas, prediction, confidence, mode, prev_point, prev_rotate_point, rotation_x, rotation_y
    with state_lock:
        canvas = np.zeros_like(canvas)
        prediction = None
        confidence = 0.0
        mode = "draw"
        prev_point = None
        prev_rotate_point = None
        rotation_x = 0.0
        rotation_y = 0.0
    return jsonify(current_state())


@app.get("/video_feed")
def video_feed():
    def stream():
        while not stop_event.is_set():
            with state_lock:
                frame = latest_frame.copy() if latest_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg_bytes(frame) + b"\r\n"
            time.sleep(1 / 24)
    return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
