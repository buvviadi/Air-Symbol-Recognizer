import os 
import cv2 
import numpy as np 
import torch 
import torch.nn as nn
from sklearn.model_selection import train_test_split
from model import SymbolCNN

DATA_DIR = "data"
IMAGE_SIZE = 64

symbols = os.listdir(DATA_DIR)
print("Symbols:", symbols)

images = []
labels = []

for label_index, symbol in enumerate(symbols):
    symbol_folder = os.path.join(DATA_DIR, symbol)
    for file_name in os.listdir(symbol_folder):
        file_path = os.path.join(symbol_folder, file_name)
        img = cv2.imread(file_path)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        images.append(img)
        labels.append(label_index)

images = np.array(images, dtype=np.float32) / 255.0
labels = np.array(labels)
images = images.transpose(0, 3, 1, 2)

X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
y_train=torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

model = SymbolCNN(num_classes=len(symbols))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 20

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs, 1)
        accuracy = (predicted == y_test).float().mean()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - test Accuracy: {accuracy.item():.4f}")

torch.save(model.state_dict(), "Symbol_model.pth")
print("Model save as symbol_model.pth")

with open("labels.txt", "w") as f:
         for symbol in symbols:
              f.write(symbol+"\n")
         print("labels saved as labels.txt")


        