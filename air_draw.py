import cv2
import mediapipe as mp 
import numpy as np 
import os 

symbol_name = input("Enter the symbol name you're about to draw: ")
save_folder =  os.path.join("data", symbol_name)
os.makedirs(save_folder, exist_ok=True)

existing_files = os.listdir(save_folder)
count = len(existing_files)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

success, frame = cap.read()
h, w, _ = frame.shape 
canvas = np.zeros ((h, w, 3), dtype=np.uint8)

prev_x, prev_y = None, None

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results =  hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_landmarks =  results.multi_hand_landmarks[0]
        index_tip = hand_landmarks.landmark[8]
        x = int(index_tip.x * w)
        y =  int(index_tip.y * h)

        if prev_x is not None:
            cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 255, 255), 5)

        prev_x, prev_y = x, y
    else:
        prev_x, prev_y = None, None

    small_frame = cv2.resize(frame, (200, 150))
    display = canvas.copy()
    display[0:150, w-200:w] = small_frame

    cv2.imshow("Air Draw", display)

    key =  cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break 
    elif key == ord('c'):
        canvas =  np.zeros((h, w, 3), dtype=np.uint8)
    elif key == ord('s'):
        file_path = os.path.join(save_folder, f"{count}.png")
        cv2.imwrite(file_path, canvas)
        count += 1
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        print(f"Saved {file_path}")

cap.release()
cv2.destroyAllWindows()