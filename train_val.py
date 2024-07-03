import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from video_dataset import VideoDataset  # Import your VideoDataset class from video_dataset.py
from model import Model  # Import your Model class from model.py

def main(dataset_path, batch_size, max_len, image_size, num_epochs, learning_rate):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    train_dataset = VideoDataset(root_dir=dataset_path, phase="train", transform=transform, n_frames=max_len)
    val_dataset = VideoDataset(root_dir=dataset_path, phase="val", transform=transform, n_frames=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    print(f"Train Dataset Length: {len(train_dataset)}, Val Dataset Length: {len(val_dataset)}")
    print(f"Train Loader Length: {len(train_loader)}, Val Loader Length: {len(val_loader)}")

    model = Model(image_size=image_size, num_frames=max_len)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            all_labels = []
            all_preds = []
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                val_loss += criterion(outputs, labels).item()

            val_loss /= len(val_loader)
            val_acc = accuracy_score(all_labels, all_preds)

        print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), 'trained_model.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViViT Video Classification - Train and Validation")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and validation")
    parser.add_argument("--max_len", type=int, default=10, help="Number of frames per video")
    parser.add_argument("--image_size", type=int, default=224, help="Image size (height and width)")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    args = parser.parse_args()
    main(args.dataset_path, args.batch_size, args.max_len, args.image_size, args.num_epochs, args.learning_rate)
