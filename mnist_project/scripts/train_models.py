import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import csv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MODEL_FOLDER = os.path.join(PROJECT_ROOT, "model")
RESULT_FOLDER = os.path.join(PROJECT_ROOT, "result")
DATA_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "data"))
LOG_FILE = os.path.join(RESULT_FOLDER, "training_results.csv")

for folder in [MODEL_FOLDER, RESULT_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(nn.Flatten(), nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 10))
    def forward(self, x): return self.main(x)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(), nn.Conv2d(32, 64, 3), nn.ReLU(),
            nn.MaxPool2d(2), nn.Flatten(), nn.Linear(9216, 128), nn.ReLU(), nn.Linear(128, 10)
        )
    def forward(self, x): return self.conv(x)

class RobustCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.Flatten(), nn.Linear(9216, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 10)
        )
    def forward(self, x): return self.conv(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(LOG_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Model_Type", "Epochs", "Trial", "Duration_Seconds"])

def run_training_session(model_class, label, epochs, trial):
    is_robust = (label == "robust_cnn")
    tform = transforms.Compose([
        transforms.RandomRotation(15) if is_robust else transforms.Lambda(lambda x: x),
        transforms.ToTensor()
    ])
    
    train_ds = datasets.MNIST(root=DATA_PATH, train=True, download=False, transform=tform)
    loader = DataLoader(train_ds, batch_size=128, shuffle=True, pin_memory=True)
    
    model = model_class().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    if device.type == 'cuda':
        torch.cuda.synchronize() 
    start_time = time.perf_counter()
    
    model.train()
    for _ in range(epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    if device.type == 'cuda':
        torch.cuda.synchronize() 
    end_time = time.perf_counter()
    
    duration = round(end_time - start_time, 4)
    
    save_path = os.path.join(MODEL_FOLDER, f"{label}_ep{epochs:02d}_t{trial}.pth")
    torch.save(model.state_dict(), save_path)
    
    with open(LOG_FILE, 'a', newline='') as f:
        csv.writer(f).writerow([label, epochs, trial, duration])
    
    print(f"Saved: {label} | Ep: {epochs} | Trial: {trial} | Time: {duration}s")

configs = [
    (MLP, "simple_mlp"),
    (CNN, "simple_cnn"),
    (RobustCNN, "robust_cnn")
]

for m_class, m_label in configs:
    for ep in range(1, 11): 
        for t in range(1, 4): 
            run_training_session(m_class, m_label, ep, t)

print(f"\nTraining complete | Logs: {LOG_FILE} | Models: {MODEL_FOLDER}")