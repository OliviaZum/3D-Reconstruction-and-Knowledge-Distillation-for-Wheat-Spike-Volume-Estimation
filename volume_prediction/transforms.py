import torch
from torch import nn
from torchvision.transforms import v2
from torchvision.transforms import functional as F

def resize_pad_transform():
    # Scales and pads to 224x224 preserving aspect ratio
    def resize_pad(x):
        w, h = x.shape[-1], x.shape[-2]
        if w > h:
            n_w = 224
            n_h = int(224 * h / w)
        else:
            n_h = 224
            n_w = int(224 * w / h)

        x = F.resize(x, (n_h, n_w))
        x = F.pad(x, (0, 0, 224 - n_w, 224 - n_h))
        return x
    return v2.Lambda(resize_pad)

def standard_transforms(augment: bool):
    return v2.Compose([
            resize_pad_transform(),
            v2.AutoAugment() if augment else v2.Identity(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])