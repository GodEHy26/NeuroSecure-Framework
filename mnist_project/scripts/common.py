"""Shared paths, model definitions, data loading and evaluation helpers.

Every script imports from here so that paths resolve relative to the
repository rather than the current working directory.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))           # mnist_project/
REPO_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")                                # downloaded by torchvision
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
VISUALS_DIR = os.path.join(PROJECT_ROOT, "visuals")
CM_DIR = os.path.join(VISUALS_DIR, "confusion_matrices")
SAMPLES_DIR = os.path.join(VISUALS_DIR, "samples")
FIGURES_DIR = os.path.join(VISUALS_DIR, "figures")

TRAINING_CSV = os.path.join(RESULTS_DIR, "training_results.csv")
ATTACK_CSV = os.path.join(RESULTS_DIR, "adversarial_attack_results.csv")

MODEL_LABELS = ["simple_mlp", "simple_cnn", "robust_cnn"]
DISPLAY_NAMES = {"simple_mlp": "Simple MLP", "simple_cnn": "Simple CNN", "robust_cnn": "Robust CNN"}
EPOCHS = list(range(1, 11))
TRIALS = [1, 2, 3]
EPSILONS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

BATCH_SIZE = 128
LEARNING_RATE = 0.001
ROTATION_DEGREES = 15


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(nn.Flatten(), nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 10))

    def forward(self, x):
        return self.main(x)


class CNN(nn.Module):
    # 3x3 kernels, stride 1, no padding: 28 -> 26 -> 24 -> maxpool -> 12; 64 * 12 * 12 = 9216
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(), nn.Conv2d(32, 64, 3), nn.ReLU(),
            nn.MaxPool2d(2), nn.Flatten(), nn.Linear(9216, 128), nn.ReLU(), nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.conv(x)


class RobustCNN(nn.Module):
    # Same conv/dense layers as CNN, plus batch norm and dropout ("robust" = standard regularization,
    # not adversarial training). Random rotation is applied to the training data only.
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.Flatten(), nn.Linear(9216, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.conv(x)


MODEL_CLASSES = {"simple_mlp": MLP, "simple_cnn": CNN, "robust_cnn": RobustCNN}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_path(label, epochs, trial):
    return os.path.join(MODEL_DIR, f"{label}_ep{epochs:02d}_t{trial}.pth")


def load_model(label, epochs, trial, device):
    """Load a saved model and put it in eval mode (fixed BN statistics, dropout off)."""
    model = MODEL_CLASSES[label]().to(device)
    model.load_state_dict(torch.load(model_path(label, epochs, trial), map_location=device))
    model.eval()
    return model


def train_dataset(label):
    # Rotation is a training-time augmentation for the Robust CNN only; test images are never rotated.
    tforms = [transforms.RandomRotation(ROTATION_DEGREES)] if label == "robust_cnn" else []
    tform = transforms.Compose(tforms + [transforms.ToTensor()])
    return datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=tform)


def test_tensors(device):
    """Return the full MNIST test set as tensors in [0, 1], shape (10000, 1, 28, 28)."""
    ds = datasets.MNIST(root=DATA_DIR, train=False, download=True)
    x = ds.data.float().div(255.0).unsqueeze(1)
    return x.to(device), ds.targets.clone().to(device)


@torch.no_grad()
def evaluate(model, x, y, batch_size=1000):
    """Return (accuracy %, mean cross-entropy loss, predictions) on tensors x, y."""
    model.eval()
    total_loss, preds = 0.0, []
    for i in range(0, len(x), batch_size):
        out = model(x[i:i + batch_size])
        total_loss += F.cross_entropy(out, y[i:i + batch_size], reduction="sum").item()
        preds.append(out.argmax(1))
    preds = torch.cat(preds)
    acc = (preds == y).float().mean().item() * 100
    return acc, total_loss / len(x), preds
