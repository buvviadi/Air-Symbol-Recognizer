import cv2
import mediapipe as mp
import numpy as np
import torch
from model import SymbolCNN

IMAGE_SIZE = 64 

with open("labels.txt", "r") as f:
    symbols = [line.strip() for line in f.readlines()]

model = SymbolCNN(num_classes = len(symbols))
model.load_state_dict(torch.load("symbol_model.pth"))
model.eval()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

success, frame = cap.read()
h, w, _ = frame.shape
canvas = np.zeros((h, w, 3), dtype=np.uint8)

prev_x, prev_y = None, None
prediction_text = "Draw something, then press 'p'"

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        index_tip = hand_landmarks.landmark[8]
        x = int(index_tip.x * w)
        y = int(index_tip.y * h)

        if prev_x is not None:
            cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 255, 255), 5)

        prev_x, prev_y = x, y
    else:
        prev_x, prev_y = None, None

    small_frame = cv2.resize(frame, (200,150))
    display = canvas.copy()
    display[0:150, w-200:w] = small_frame
    cv2.imshow("predict", display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        predictions_text = "Draw something, then press 'p'"
    elif key == ord('p'):
        img  = cv2.resize(canvas, (IMAGE_SIZE, IMAGE_SIZE))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img_tensor = torch.tensor(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)
            predicted_label = symbols[predicted.item()]

        prediction_text = f"Prediction: {predicted_label}"
        print(prediction_text)

cap.release()
cv2.destroyAllWindows
