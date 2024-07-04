import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

from video_dataset import VideoDataset  # Import your VideoDataset class from video_dataset.py
from model import Model  # Import your Model class from model.py

def validate(model, criterion, val_loader, device):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total

        # Calculate precision, recall, and F1 score
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )

    return val_loss, val_accuracy, val_precision, val_recall, val_f1

def main(dataset_path, batch_size, max_len, image_size, num_epochs, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # Load datasets and create data loaders
    train_dataset = VideoDataset(root_dir=dataset_path, phase="train", transform=transform, n_frames=max_len)
    val_dataset = VideoDataset(root_dir=dataset_path, phase="val", transform=transform, n_frames=max_len)
    test_dataset = VideoDataset(root_dir=dataset_path, phase="test", transform=transform, n_frames=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Train Dataset Length: {len(train_dataset)}, Val Dataset Length: {len(val_dataset)}, Test Dataset Length: {len(test_dataset)}")
    print(f"Train Loader Length: {len(train_loader)}, Val Loader Length: {len(val_loader)}, Test Loader Length: {len(test_loader)}")

    # Initialize model, loss function, and optimizer
    model = Model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation after each epoch
        val_loss, val_accuracy, val_precision, val_recall, val_f1 = validate(model, criterion, val_loader, device)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_accuracy:.4f}, "
              f"Val Precision: {val_precision:.4f}, "
              f"Val Recall: {val_recall:.4f}, "
              f"Val F1: {val_f1:.4f}")

    # Testing
    model.eval()
    test_loss, test_accuracy, test_precision, test_recall, test_f1 = validate(model, criterion, test_loader, device)

    print(f"Test Loss: {test_loss:.4f}, "
          f"Test Acc: {test_accuracy:.4f}, "
          f"Test Precision: {test_precision:.4f}, "
          f"Test Recall: {test_recall:.4f}, "
          f"Test F1: {test_f1:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), 'trained_model.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViViT Video Classification")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--max_len", type=int, default=10, help="Maximum number of frames per video")
    parser.add_argument("--image_size", type=int, default=224, help="Image size (height and width)")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    args = parser.parse_args()

    main(args.dataset_path, args.batch_size, args.max_len, args.image_size, args.num_epochs, args.learning_rate)
