import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import ViTConfig, ViTForVideoClassification

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
            # Opens each frame and converts it to RGB format
            image = Image.open(frame_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)

        # Stack images along new dimension (sequence length) (T, C, H, W)
        data = torch.stack(images, dim=0)

        # Rearrange to have the shape (C, T, H, W)
        data = data.permute(1, 0, 2, 3)
        return data, label

# Constants and hyperparameters
BATCH_SIZE = 16
MAX_LEN = 15
IMAGE_SIZE = 720
NUM_CLASSES = 2
NUM_FRAMES = 15
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

# Define transforms
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# Load dataset
train_dataset = VideoDataset(root_dir='/mnt/d/tienvo/dataset', phase="train", transform=transform, n_frames=MAX_LEN)
test_dataset = VideoDataset(root_dir='/mnt/d/tienvo/dataset', phase="test", transform=transform, n_frames=MAX_LEN)

# Count number of CPUs
num_workers = os.cpu_count()
print(f"Number of CPUs: {num_workers}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=num_workers, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=num_workers, shuffle=False)

# Define model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.vit = ViTForVideoClassification.from_pretrained('google/vit-base-patch32')

    def forward(self, x):
        x = x.view(-1, NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE)  # Reshape input to (batch_size, num_frames, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)  # Permute to (batch_size, channels, num_frames, height, width)
        logits = self.vit(x).logits
        return logits

# Initialize model, optimizer, and loss function
model = Model()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(NUM_EPOCHS):
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
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss:.4f}, Test Acc: {accuracy:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')
print("Model saved successfully.")
