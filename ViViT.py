import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import VivitConfig, VivitForVideoClassification

# Define VideoDataset class
class VideoDataset(Dataset):
    def __init__(self, root_dir, phase="train", transform=None, n_frames=None):
        """
        Args:
        root_dir (string): Directory with all the videos (each video as a subdirectory of frames).
        transform (callable, optional): Optional transform to be applied on a sample.
        n_frames (int, optional): Number of frames to sample from each video, uniformly. If None, use all frames.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.n_frames = n_frames
        self.phase = phase
        self.videos, self.labels = self._load_videos()

    def _load_videos(self):
        videos, labels = [], []
        class_id = 0

        video_folders = os.listdir(os.path.join(self.root_dir, self.phase))

        for folder in video_folders:
            video_paths = os.listdir(os.path.join(self.root_dir, self.phase, folder))

            for video_path in video_paths:
                video_folder = os.path.join(self.root_dir, self.phase, folder, video_path)
                # Lists all files in the video folder.
                # Sorts the frames based on the numeric part of their filenames.
                # This is crucial to ensure the frames are in the correct temporal order.
                frames = sorted(
                    (os.path.join(video_folder, f) for f in os.listdir(video_folder)),
                    key=lambda f: int("".join(filter(str.isdigit, os.path.basename(f)))),
                )

                if self.n_frames:
                    frames = self._uniform_sample(frames, self.n_frames)

                videos.append(frames)
                labels.append(class_id)

            class_id += 1

        return videos, labels

    def _uniform_sample(self, frames, n_frames):
        stride = max(1, len(frames) // n_frames)
        sampled = [frames[i] for i in range(0, len(frames), stride)]
        return sampled[:n_frames]

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_frames = self.videos[idx]
        label = self.labels[idx]
        images = []
        for frame_path in video_frames:
            #Opens each frame and converts it to RGB format
            image = Image.open(frame_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)

        # Stack images along new dimension (sequence length) (T, C, H, W)
        data = torch.stack(images, dim=0)

        # Rearrange to have the shape (C, T, H, W)
        data = data.permute(1, 0, 2, 3)
        return data, label

# Define Model class
class Model(nn.Module):
    def __init__(self, num_classes=2, image_size=720, num_frames=15):
        super(Model, self).__init__()
        cfg = VivitConfig(
            num_classes=num_classes,
            image_size=image_size,
            num_frames=num_frames,
            patch_size = 16,
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=12,
            intermediate_size=3072,
            dropout_rate=0.1,
            initializer_range=0.02
        )

        self.vivit = VivitForVideoClassification(config=cfg)

    def forward(self, x_3d):
        x_3d = x_3d.permute(0, 2, 1, 3, 4)  # Ensure the input is in the shape (B, C, T, H, W)
        out = self.vivit(x_3d)
        return out.logits

def main():
    # Parse arguments from command line
    parser = argparse.ArgumentParser(description='Train Video Classification Model')
    parser.add_argument('--root_dir', type=str, required=True,
                        help='Root directory of the dataset containing train and test folders')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training (default: 16)')
    parser.add_argument('--max_len', type=int, default=15,
                        help='Maximum number of frames per video (default: 15)')
    parser.add_argument('--image_size', type=int, default=720,
                        help='Image size (both height and width) after resize (default: 720)')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs for training (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    args = parser.parse_args()

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    # Load dataset
    train_dataset = VideoDataset(root_dir=args.root_dir, phase="train", transform=transform, n_frames=args.max_len)
    test_dataset = VideoDataset(root_dir=args.root_dir, phase="test", transform=transform, n_frames=args.max_len)

    # Count number of CPUs
    cpus = os.cpu_count()
    print(f"Number of CPUs: {cpus}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)

    # Create an instance of the model
    model = Model()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                test_loss += criterion(outputs, labels).item()

            test_loss /= len(test_loader)
            accuracy = correct / total

        # Print progress
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss:.4f}, Test Acc: {accuracy:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), 'trained_model.pth')
    print("Model saved successfully.")

    # Example inference (optional)
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            for i in range(len(predicted)):
                print(f"Actual: {labels[i]}, Predicted: {predicted[i]}")

if __name__ == '__main__':
    main()
