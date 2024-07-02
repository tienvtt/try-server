import torch
import torch.nn as nn
from transformers import ViTModel

class VideoViT(nn.Module):
    def __init__(self, num_classes, num_frames=15):
        super(VideoViT, self).__init__()
        self.num_frames = num_frames
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.fc = nn.Linear(self.vit.config.hidden_size * self.num_frames, num_classes)

    def forward(self, frames):
        # Reshape frames to batch size x num_frames x channels x height x width
        batch_size, _, channels, height, width = frames.shape
        frames = frames.view(batch_size * self.num_frames, channels, height, width)

        # Feed frames through ViT
        outputs = self.vit(frames)

        # Flatten and pass through FC layer
        out = outputs.last_hidden_state.view(batch_size, self.num_frames * self.vit.config.hidden_size)
        out = self.fc(out)
        return out
# Define dataset and DataLoader
train_dataset = VideoDataset(root_dir='/mnt/d/tienvo/dataset', phase="train", transform=transform, n_frames=MAX_LEN)
test_dataset = VideoDataset(root_dir='/mnt/d/tienvo/dataset', phase="test", transform=transform, n_frames=MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize model, optimizer, and loss function
model = VideoViT(num_classes=2, num_frames=MAX_LEN)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    for frames, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Evaluation loop
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        correct = 0
        total = 0
        for frames, labels in test_loader:
            outputs = model(frames)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loss += criterion(outputs, labels).item()

        test_loss /= len(test_loader)
        accuracy = correct / total

    # Print progress
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss:.4f}, Test Acc: {accuracy:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'video_vit_model.pth')
