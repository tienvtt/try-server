import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import VivitConfig, VivitForVideoClassification

class VideoDataset(Dataset):
    def __init__(self, root_dir, phase="train", transform=None, n_frames=None):
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
                frames = sorted(
                    (os.path.join(video_folder, f) for f in os.listdir(video_folder)),
                    key=lambda f: int("".join(filter(str.isdigit, os.path.basename(f))))
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
            image = Image.open(frame_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)
        data = torch.stack(images, dim=0)
        data = data.permute(1, 0, 2, 3)
        return data, label

class Model(nn.Module):
    def __init__(self, num_classes=2, image_size=224, num_frames=10):
        super(Model, self).__init__()
        cfg = VivitConfig(
            num_classes=num_classes,
            image_size=image_size,  # Ensure this is an int
            num_frames=num_frames,
            patch_size=16,
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=12,
            intermediate_size=3072,
            dropout_rate=0.1,
            initializer_range=0.02
        )
        self.vivit = VivitForVideoClassification(config=cfg)

    def forward(self, x_3d):
        x_3d = x_3d.permute(0, 2, 1, 3, 4)
        out = self.vivit(x_3d)
        return out.logits

def main(dataset_path):
    BATCH_SIZE = 4  # Reduced batch size
    MAX_LEN = 10  # Reduced number of frames
    IMAGE_SIZE = 224  # Ensure this is an int
    num_epochs = 50
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    train_dataset = VideoDataset(root_dir=dataset_path, phase="train", transform=transform, n_frames=MAX_LEN)
    test_dataset = VideoDataset(root_dir=dataset_path, phase="test", transform=transform, n_frames=MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)
    model = Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
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
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss:.4f}, Test Acc: {accuracy:.4f}")
    torch.save(model.state_dict(), 'trained_model.pth')
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            for i in range(len(predicted)):
                print(f"Actual: {labels[i]}, Predicted: {predicted[i]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViViT Video Classification")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset directory")
    args = parser.parse_args()
    main(args.dataset_path)
