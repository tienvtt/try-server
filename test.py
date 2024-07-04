import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

from video_dataset import VideoDataset  # Import your VideoDataset class from video_dataset.py
from model import Model  # Import your Model class from model.py

def test(model, criterion, test_loader, device):
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        test_loss /= len(test_loader)
        test_accuracy = test_correct / test_total

        # Calculate precision, recall, and F1 score
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )

    return test_loss, test_accuracy, test_precision, test_recall, test_f1

def main(dataset_path, batch_size, max_len, image_size, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # Load test dataset and create data loader
    test_dataset = VideoDataset(root_dir=dataset_path, phase="test", transform=transform, n_frames=max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Test Dataset Length: {len(test_dataset)}, Test Loader Length: {len(test_loader)}")

    # Initialize model and load pretrained weights
    model = Model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    criterion = nn.CrossEntropyLoss()

    # Testing
    test_loss, test_accuracy, test_precision, test_recall, test_f1 = test(model, criterion, test_loader, device)

    print(f"Test Loss: {test_loss:.4f}, "
          f"Test Acc: {test_accuracy:.4f}, "
          f"Test Precision: {test_precision:.4f}, "
          f"Test Recall: {test_recall:.4f}, "
          f"Test F1: {test_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViViT Video Classification Test")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for testing")
    parser.add_argument("--max_len", type=int, default=10, help="Maximum number of frames per video")
    parser.add_argument("--image_size", type=int, default=224, help="Image size (height and width)")
    parser.add_argument("model_path", type=str, help="Path to the trained model state_dict.pth file")
    args = parser.parse_args()

    main(args.dataset_path, args.batch_size, args.max_len, args.image_size, args.model_path)
