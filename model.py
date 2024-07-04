import torch
import torch.nn as nn
from transformers import VivitConfig, VivitForVideoClassification

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
