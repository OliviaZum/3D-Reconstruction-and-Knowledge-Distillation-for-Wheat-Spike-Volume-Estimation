from typing import List, Tuple
import warnings
import torch
from torch.utils import data
import os
import json
from torchvision.transforms import v2
from torchvision.io import read_image
import joblib
import tqdm
import random

class CutSpikeSingleImageDataset(data.Dataset):
    def __init__(self, base_folder: str, split: str, manual=True, transforms = v2.Identity, cache_file: str = None) -> None:
        super().__init__()

        self.spike_image_names = []
        self.volumes = []

        t = "manual" if manual else "automatic"
        spikes = [os.path.join(base_folder, split, x) for x in os.listdir(os.path.join(base_folder, split))]
        for spike in spikes:
            img_folder = os.path.join(spike, t)
            img_names = [os.path.join(img_folder, x) for x in os.listdir(img_folder)]
            self.spike_image_names.append(img_names)
            with open(os.path.join(spike, "data.json")) as f:
                volume = json.load(f)["volume"]
            self.volumes.append(volume)

        self.image_index_to_spike_image = {}
        self.spike_to_image_indexes = {}
        c = 0
        for i, spike_img_names in enumerate(self.spike_image_names):
            self.spike_to_image_indexes[i] = []
            for j in range(len(spike_img_names)):
                self.image_index_to_spike_image[c] = (i, j)
                self.spike_to_image_indexes[i].append(c)
                c += 1

        self.transforms = transforms

        if cache_file:
            self.using_cache = True
            self.images_cache = joblib.load(cache_file)
            u = self.images_cache
            if u["base_folder"] != base_folder or u["split"] != split or u["manual"] != manual:
                warnings.warn("The loaded cache file does not aggree with your configuration of the datasource. Maybe check if this contains what you expected")
        else:
            self.images_cache = {}
            self.images_cache["base_folder"] = base_folder
            self.images_cache["split"] = split
            self.images_cache["manual"] = manual
            self.images_cache["images"] = None
            self.using_cache = False

    def create_image_cache(self, num_iters: int, filename: str):
        with torch.no_grad():
            self.images_cache["images"] = []
            dl = data.DataLoader(self, 500, False)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            feature_extractor = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval().to(device)
            for i in range(num_iters):
                current_list = []
                for img, _ in tqdm.tqdm(dl, total=len(self) // dl.batch_size):
                    img = img.to(device)
                    features = feature_extractor(img)
                    features = features.cpu()
                    current_list.append(features)
                current_list = torch.concat(current_list, dim=0)
                self.images_cache["images"].append(current_list)
            self.images_cache["images"] = torch.stack(self.images_cache["images"])

            joblib.dump(self.images_cache, filename)

    def __len__(self):
        return len(self.image_index_to_spike_image)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, float]:
        spike, idx = self.image_index_to_spike_image[index]
        volume = self.volumes[spike]
        if self.using_cache:
            r = random.randint(0, self.images_cache["images"].shape[0] -1)
            img = self.images_cache["images"][r, index]
        else:
            img_name = self.spike_image_names[spike][idx]
            img = read_image(img_name)
            img = self.transforms(img)

        return img, torch.tensor(volume, dtype=torch.float32)

class CutSpikeDataset(CutSpikeSingleImageDataset):
    def __init__(self, base_folder: str, split: str, manual=True, transforms = v2.Identity(), cache_file = None) -> None:
        super().__init__(base_folder, split, manual, transforms, cache_file)

    def __len__(self):
        return len(self.spike_image_names)

    def __getitem__(self, index) -> Tuple[List[torch.Tensor], float]:
        max_seq_len = 12
        min_seq_len = 6

        indexes = self.spike_to_image_indexes[index]
        imgs = []
        for j in indexes:
            img, volume = super().__getitem__(j)
            imgs.append(img)

        

        return imgs, torch.tensor(volume, dtype=torch.float32)
    