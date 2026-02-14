import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from models.msca_pfr_model import MobileNetV2_MSCA_PFR
from utils.dataset_loader import ImageFolderDataset
from utils.metrics import accuracy_score
from utils.plot_tools import plot_training_curves


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():

    # -----------------------------
    # Load configuration file
    # -----------------------------
    config = load_config("config.yaml")

    train_dir = config["paths"]["train_dir"]
    val_dir = config["paths"]["val_dir"]
    checkpoint_dir = config["paths"]["checkpoint_dir"]

    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    lr = config["training"]["learning_rate"]
    num_workers = config["training"]["num_workers"]
    patience = config["training"]["patience"]

    num_classes = config["model"]["num_classes"]

    # -----------------------------
    # Device setup
    # -----------------------------
    use_gpu = config["device"]["use_gpu"] and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    print(f"\nUsing device: {device}\n")

    # -----------------------------
    # Create checkpoint folder
    # -----------------------------
    os.makedirs(checkpoint_dir, exist_ok=True)

    # -----------------------------
    # Data Transforms
    # -----------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # -----------------------------
    # Load Datasets
    # -----------------------------
    train_dataset = ImageFolderDataset(train_dir, transform)
    val_dataset = ImageFolderDataset(val_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)

    # -----------------------------
    # Initialize Model
    # -----------------------------
    model = MobileNetV2_MSCA_PFR(num_classes=num_classes).to(device)

    # -----------------------------
    # Loss & Optimizer
    # -----------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # -----------------------------
    # Training Loop
    # -----------------------------
    best_val_acc = 0.0
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    print("\nStarting Training...\n")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_train_loss = 0
        epoch_train_correct = 0
        epoch_train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            epoch_train_correct += (preds == labels).sum().item()
            epoch_train_total += labels.size(0)

        train_loss = epoch_train_loss / len(train_loader)
        train_acc = 100 * epoch_train_correct / epoch_train_total

        # -----------------------------
        # Validation
        # -----------------------------
        model.eval()
        epoch_val_loss = 0
        epoch_val_correct = 0
        epoch_val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                epoch_val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                epoch_val_correct += (preds == labels).sum().item()
                epoch_val_total += labels.size(0)

        val_loss = epoch_val_loss / len(val_loader)
        val_acc = 100 * epoch_val_correct / epoch_val_total

        # Store Metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}  Val Acc:   {val_acc:.2f}%\n")

        # -----------------------------
        # Early Stopping + Checkpoint
        # -----------------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
            print("  → Saved new best model\n")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("\nEarly stopping triggered.\n")
                break

    print("Training Complete.\n")

    # -----------------------------
    # Save training curves
    # -----------------------------
    plot_training_curves(train_losses, val_losses,
                         train_accuracies, val_accuracies,
                         os.path.join(checkpoint_dir, "training_curves.png"))
