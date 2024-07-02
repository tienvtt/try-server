import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import VivitConfig, VivitForVideoClassification

import argparse

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_len=15, image_size=224):
        self.root_dir = root_dir
        self.transform = transform
        self.max_len = max_len
        self.image_size = image_size
        self.data = []
        self.labels = []
        
        # Loading data
        for label in ['normal', 'restricted']:
            for video_folder in os.listdir(os.path.join(root_dir, label)):
                frames = []
                frame_folder = os.path.join(root_dir, label, video_folder)
                for frame in os.listdir(frame_folder):
                    frame_path = os.path.join(frame_folder, frame)
                    frames.append(frame_path)
                self.data.append(frames)
                self.labels.append(0 if label == 'normal' else 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frames = self.data[idx]
        label = self.labels[idx]
        
        processed_frames = []
        for frame_path in frames:
            try:
                image = Image.open(frame_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                processed_frames.append(image)
            except Exception as e:
                print(f"Error processing file {frame_path}: {e}")

        # Padding or truncating the frames to max_len
        if len(processed_frames) < self.max_len:
            processed_frames.extend([torch.zeros(3, self.image_size, self.image_size) for _ in range(self.max_len - len(processed_frames))])
        else:
            processed_frames = processed_frames[:self.max_len]
        
        return torch.stack(processed_frames), torch.tensor(label)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_len', type=int, default=15)
    parser.add_argument('--image_size', type=int, default=720)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    train_dataset = CustomDataset(root_dir=args.root_dir, transform=transform, max_len=args.max_len, image_size=args.image_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    config = VivitConfig(num_frames=args.max_len, image_size=args.image_size)
    model = VivitForVideoClassification(config)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

    print('Training finished.')

if __name__ == "__main__":
    main()
