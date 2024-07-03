import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import ViTConfig, ViTModel

# Define VideoDataset class
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
            image = Image.open(frame_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)

        data = torch.stack(images, dim=0)
        data = data.permute(1, 0, 2, 3)
        return data, label

# Model definition
class VideoClassificationModel(nn.Module):
    def __init__(self, num_classes=2, image_size=720, num_frames=15):
        super(VideoClassificationModel, self).__init__()
        self.config = ViTConfig(
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=12,
            intermediate_size=3072,
            dropout_rate=0.1,
            initializer_range=0.02
        )

        self.vit = ViTModel(config=self.config)

        # Additional layers for classification
        self.fc = nn.Linear(self.config.hidden_size, num_classes)

    def forward(self, x):
        x = x.view(-1, num_frames, 3, image_size, image_size)  # Reshape input to (batch_size, num_frames, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)  # Permute to (batch_size, channels, num_frames, height, width)
        outputs = self.vit(x)
        last_hidden_state = outputs.last_hidden_state
        logits = self.fc(last_hidden_state[:, 0])  # Use only the first token's output for classification
        return logits

# Hyperparameters
BATCH_SIZE = 16
MAX_LEN = 15
IMAGE_SIZE = 720
num_epochs = 50

# Data transformations
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# Load datasets
train_dataset = VideoDataset(root_dir='/mnt/d/tienvo/dataset', phase="train", transform=transform, n_frames=MAX_LEN)
test_dataset = VideoDataset(root_dir='/mnt/d/tienvo/dataset', phase="test", transform=transform, n_frames=MAX_LEN)

# Count number of CPUs for DataLoader
cpus = os.cpu_count()
print(f"Number of CPUs: {cpus}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=cpus, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=cpus, shuffle=False)

# Initialize model, optimizer, and loss function
model = VideoClassificationModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
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
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss:.4f}, Test Acc: {accuracy:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')

# Final evaluation
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        for i in range(len(predicted)):
            print(f"Actual: {labels[i]}, Predicted: {predicted[i]}")
