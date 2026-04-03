import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load DINO
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
model = model.to(device)
model.eval()

# ---- Load ONE image manually ----
img_path = "/projects/zumstego/volume_prediction_fip/Boxes-ds/segmented_distance_depth/6_.jpg"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

img_pil = Image.open(img_path).convert("RGB")
img_tensor = transform(img_pil).unsqueeze(0).to(device)