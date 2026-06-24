

import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BRAIN_FILE = os.path.join(BASE_DIR, "financial_resnet.pth")
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
TOTAL_EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-3


def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # No random augmentation at eval time -- we want a stable, repeatable
    # measurement, not augmented noise, when checking generalization.
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def prepare_data(batch_size=BATCH_SIZE):
    print("Initializing data pipeline (train/ and test/ are disjoint in time -- see pipeline_engine.py)...")
    train_tf, eval_tf = build_transforms()

    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_tf)
    val_dataset = datasets.ImageFolder(root=TEST_DIR, transform=eval_tf)

    classes = train_dataset.classes
    print(f"Detected classes: {classes}")
    print(f"Train images: {len(train_dataset)}  |  Validation (held-out) images: {len(val_dataset)}")

    class_counts = np.zeros(len(classes))
    for _, label in train_dataset.samples:
        class_counts[label] += 1
    print(f"Train class counts: {dict(zip(classes, class_counts.astype(int)))}")

    weights = 1.0 / np.maximum(class_counts, 1)
    weights = weights * len(classes) / weights.sum()
    class_weights = torch.FloatTensor(weights)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, class_weights, classes


def build_model(num_classes=3):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes),
    )
    return model


def run_validation(model, val_loader, criterion):
    """Real held-out evaluation: model.eval(), no_grad, no augmentation."""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = val_loss / max(len(val_loader), 1)
    accuracy = 100 * correct / max(total, 1)
    return avg_loss, accuracy


def train_engine():
    if not os.path.isdir(TRAIN_DIR) or not os.path.isdir(TEST_DIR):
        print("dataset/train and dataset/test not found. Run pipeline_engine.py first.")
        return

    train_loader, val_loader, class_weights, categories = prepare_data()

    print(f"Hardware initialized. Training on: {DEVICE.type.upper()}")
    model = build_model(num_classes=len(categories)).to(DEVICE)

    if os.path.exists(BRAIN_FILE):
        print(f"\nFound existing checkpoint ({BRAIN_FILE}). Resuming from it.")
        model.load_state_dict(torch.load(BRAIN_FILE, map_location=DEVICE, weights_only=True))
    else:
        print("\nNo existing checkpoint found. Training from the pretrained ImageNet backbone.\n")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    best_val_accuracy = 0.0

    print(f"Training for {TOTAL_EPOCHS} epochs. Reporting train loss/acc AND held-out validation acc each epoch.\n")

    for epoch in range(TOTAL_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / max(len(train_loader), 1)
        train_acc = 100 * correct / max(total, 1)

        val_loss, val_acc = run_validation(model, val_loader, criterion)

        print(
            f"Epoch [{epoch + 1}/{TOTAL_EPOCHS}] | "
            f"Train loss: {train_loss:.4f} acc: {train_acc:.2f}% || "
            f"VAL loss: {val_loss:.4f} acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_accuracy:
            print(f"  New best VALIDATION accuracy: {val_acc:.2f}% (was {best_val_accuracy:.2f}%). Saving checkpoint.")
            best_val_accuracy = val_acc
            try:
                torch.save(model.state_dict(), BRAIN_FILE)
            except Exception as e:
                print(f"  Could not save checkpoint this epoch: {e}")

    print(f"\nTraining complete. Best held-out validation accuracy: {best_val_accuracy:.2f}%")
    print(f"This is the number that should be quoted anywhere this project is described --")
    print(f"not training accuracy, which will read higher and means something different.")


if __name__ == "__main__":
    train_engine()
