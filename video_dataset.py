import os
from PIL import Image
from torch.utils.data import Dataset

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
