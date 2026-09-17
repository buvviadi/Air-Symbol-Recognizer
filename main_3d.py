import cv2
import mediapipe as mp
import numpy as np
import torch
from model import SymbolCNN
from ursina import *

IMAGE_SIZE = 64

with open("labels.txt", "r") as f:
    symbols_list = [line.strip() for line in f.readlines()]

model = SymbolCNN(num_classes=len(symbols_list))
model.load_state_dict(torch.load("symbol_model.pth"))
model.eval()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(0)

success, frame = cap.read()
h, w, _ = frame.shape
canvas = np.zeros((h, w, 3), dtype=np.uint8)

prev_x, prev_y = None, None
prev_hand_x = None
mode = "draw"

app = Ursina()
window.color = color.rgb(240, 243, 246)
window.title = "Air-Draw 3D Inspector"

camera.position = (0, 0, -15)

card_bg = Entity(parent=camera.ui, model='quad', color=color.rgba(255, 255, 255, 220), scale=(0.5, 0.4), position=(0.5, 0.2))
title_text = Text(parent=card_bg, text="COMPONENT", position=(-0.45, 0.4), scale=1.2, color=color.black)
info_text = Text(parent=card_bg, text="Draw a symbol, then press P", position=(-0.45, 0.2), scale=0.9, color=color.dark_gray)

components_data = {
    "resistor": {
        "title": "AXIAL RESISTOR",
        "lines": ["Value: 220 Ohm +/-5%", "Type: Carbon Film", "Power: 0.25W", "Function: Limits current flow"],
    },
    "gear": {
        "title": "MECHANICAL GEAR",
        "lines": ["Teeth: 12", "Material: Steel", "Function: Transmits rotational force"],
    },
    "pendulum": {
        "title": "SIMPLE PENDULUM",
        "lines": ["Length: 1.0 m", "Period: ~2.0 s", "Function: Demonstrates harmonic motion"],
    },
    "angle": {
        "title": "GEOMETRIC ANGLE",
        "lines": ["Measured in: Degrees", "Function: Space between two lines"],
    },
}

card_border = Entity(parent=camera.ui, model='quad', color=color.cyan, scale=(0.54, 0.44), position=(0.5, 0.2))
card_bg = Entity(parent=camera.ui, model='quad', color=color.rgba(255, 255, 255, 220), scale=(0.5, 0.4), position=(0.5, 0.2))
title_text = Text(parent=card_bg, text="COMPONENT", position=(-0.45, 0.4), scale=1.2, color=color.black)
info_text = Text(parent=card_bg, text="Draw a symbol, then press P", position=(-0.45, 0.2), scale=0.9, color=color.dark_gray)

item_holder = Entity(position=(0, 0, 0))
debug_cube = Entity(parent=item_holder, model='cube', color=color.red, scale=3)

resistor_group = Entity(parent=item_holder, enabled=False)
Entity(parent=resistor_group, model=Cylinder(radius=0.3, height=2, direction=(1,0,0)), color=color.rgb(210, 180, 140))

band_positions = [-0.6, -0.2, 0.2, 0.6]
band_colors = [color.red, color.black, color.gold, color.gold]
for pos, band_color in zip(band_positions, band_colors):
    Entity(parent=resistor_group, model=Cylinder(radius=0.32, height=0.15, direction=(1,0,0)), color=band_color, x=pos)

Entity(parent=resistor_group, model=Cylinder(radius=0.05, height=1.5, direction=(1,0,0)), color=color.light_gray, x=-1.75)
Entity(parent=resistor_group, model=Cylinder(radius=0.05, height=1.5, direction=(1,0,0)), color=color.light_gray, x=1.75)

gear_group = Entity(parent=item_holder, enabled=True)
Entity(parent=gear_group, model=Cylinder(resolution=12), color=color.rgb(100, 110, 120), scale=(2.2, 0.2, 2.2))
Entity(parent=gear_group, model='cube', color=color.rgb(80, 90, 100), scale=(0.6, 0.3, 2.6))
Entity(parent=gear_group, model='cube', color=color.rgb(80, 90, 100), scale=(2.6, 0.3, 0.6))

pendulum_group = Entity(parent=item_holder, enabled=False)
Entity(parent=pendulum_group, model=Cylinder(resolution=8), color=color.dark_gray, scale=(0.05, 3, 0.05), y=1.5)
Entity(parent=pendulum_group, model='sphere', color=color.rgb(50, 100, 200), scale=(0.8, 0.8, 0.8), y=-0.2)

angle_group = Entity(parent=item_holder, enabled=False)
Entity(parent=angle_group, model='cube', color=color.rgb(200, 80, 150), scale=(2, 0.05, 0.05), x=1)
Entity(parent=angle_group, model='cube', color=color.rgb(80, 150, 200), scale=(0.05, 2, 0.05), y=1)

shape_groups = {
    "resistor": resistor_group,
    "gear": gear_group,
    "pendulum": pendulum_group,
    "angle": angle_group,
}

def show_symbol(name):
    for group in shape_groups.values():
        group.enabled = False
    if name in shape_groups:
        shape_groups[name].enabled = True
        data = components_data[name]
        title_text.text = data["title"]
        info_text.text = "\n".join(data["lines"])

def predict_drawing():
    img = cv2.resize(canvas, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img_tensor = torch.tensor(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    return symbols_list[predicted.item()]

def input(key):
    global canvas, mode
    if key == 'p':
        label = predict_drawing()
        show_symbol(label)
        mode = "rotate"
        print("Predicted:", label)
    elif key == 'c':
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        mode = "draw"
        title_text.text = "COMPONENT"
        info_text.text = "Draw a symbol, then press P"
        for group in shape_groups.values():
            group.enabled = False

def update():
    global prev_x, prev_y, prev_hand_x, canvas

    success, frame = cap.read()
    if not success:
        return

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        index_tip = hand_landmarks.landmark[8]
        x = int(index_tip.x * w)
        y = int(index_tip.y * h)

        if mode == "draw":
            if prev_x is not None:
                cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 255, 255), 5)
            prev_x, prev_y = x, y
        else:
            if prev_hand_x is not None:
                delta = x - prev_hand_x
                item_holder.rotation_y += delta * 0.5
            prev_hand_x = x
            prev_x, prev_y = None, None
    else:
        prev_x, prev_y = None, None
        prev_hand_x = None

    display = canvas.copy()
    small_frame = cv2.resize(frame, (200, 150))
    display[0:150, w-200:w] = small_frame

    mode_text = "DRAW MODE - draw, then press P" if mode == "draw" else "ROTATE MODE - move hand left/right | C to redraw"
    cv2.putText(display, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Air Draw", display)
    cv2.waitKey(1)

app.run()

cap.release()
cv2.destroyAllWindows()