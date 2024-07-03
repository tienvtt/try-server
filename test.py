import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from video_dataset import VideoDataset  # Import your VideoDataset class from video_dataset.py
from model import Model  # Import your Model class from model.py

def main(dataset_path, batch_size, max_len, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    test_dataset = VideoDataset(root_dir=dataset_path, phase="test", transform=transform, n_frames=max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    print(f"Test Dataset Length: {len(test_dataset)}, Test Loader Length: {len(test_loader)}")

    model = Model(image_size=image_size, num_frames=max_len)
    model.load_state_dict(torch.load('trained_model.pth'))  # Load the trained model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        test_loss = 0.0
        all_labels = []
        all_preds = []
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            test_loss += criterion(outputs, labels).item()

        test_loss /= len(test_loader)
        test_acc = accuracy_score(all_labels, all_preds)
        test_precision = precision_score(all_labels, all_preds, average='macro')
        test_recall = recall_score(all_labels, all_preds, average='macro')
        test_f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test Precision: {test_precision:.4f}, "
          f"Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViViT Video Classification - Test")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for testing")
    parser.add_argument("--max_len", type=int, default=10, help="Number of frames per video")
    parser.add_argument("--image_size", type=int, default=224, help="Image size (height and width)")
    args = parser.parse_args()
    main(args.dataset_path, args.batch_size, args.max_len, args.image_size)
