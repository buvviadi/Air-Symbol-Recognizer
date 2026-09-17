import os
import cv2
import numpy as np

DATA_DIR = "data"
IMAGE_SIZE = 64

symbols = os.listdir(DATA_DIR)
print("Found symbols:", symbols)

images = []
labels = []

for label_index, symbol in enumerate(symbols):
    symbol_folder = os.path.join(DATA_DIR, symbol)
    files = os.listdir(symbol_folder)

    for file_name in files:
        file_path =  os.path.join(symbol_folder, file_name)
        img = cv2.imread(file_path)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        images.append(img)
        labels.append(label_index)

images = np.array(images)
labels = np.array(labels)

print("Total images loaded:", len(images))
print("Image array shape:", images.shape)
print("labels array shape:", labels.shape)