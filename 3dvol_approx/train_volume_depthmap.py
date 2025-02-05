import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import pandas as pd
import datasets_3d
import utils_3d
import models_3d
from pathlib import Path
import tqdm
import pointnet_utils
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def evaluate(model, dataloader, scatter=False):
    model = model.eval()
    with torch.no_grad():
        preds = []
        vs = []
        for _, batch, volume in dataloader:
            batch = batch.to(device)
            volume = volume.to(device)
            pred = model(batch)
            preds.append(pred.cpu())
            vs.append(volume.cpu())
        preds = torch.concat(preds).squeeze()
        vs = torch.concat(vs)
        preds = datasets_3d.DepthMapDataset.vol_unorm(preds)
        vs = datasets_3d.DepthMapDataset.vol_unorm(vs)
        mae = torch.mean(torch.abs(preds - vs))
        if scatter:
            plt.scatter(vs.numpy(), preds.numpy())
            plt.plot((2000, 9000), (2000, 9000))
            plt.show()
    return mae

class PointNetRegressor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pnet = pointnet_utils.PointNetEncoder(channel=6)
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = x.permute((0, 2, 1))
        x, trans, _ = self.pnet(x)
        pred = self.fc(x).squeeze()
        if self.training:
            return pred, trans, x
        else:
            return pred

def train(folder : Path, split_folder: Path, out: Path):
    val_ds = datasets_3d.DepthMapDataset(folder, folder / split_folder / "mapping_val.json", augment=False)
    val_ds.create_or_load_cache(Path(r"3dvol_approx\local_stuff\dmap_cache_val.pth"), force_recompute=False)
    val_loader = DataLoader(val_ds, 8, False)
    #model = torch.load(out, weights_only=False)
    #evaluate(model, val_loader)
    #exit(0)

    model = PointNetRegressor().to(device)
    decoder = models_3d.PCNDecoder(latent_size=1024).to(device)
    train_ds = datasets_3d.DepthMapDataset(folder, folder / split_folder / "mapping_train.json", min_seq_len=5)
    train_ds.create_or_load_cache(Path(r"3dvol_approx\local_stuff\dmap_cache_train.pth"), force_recompute=False)
    train_loader = DataLoader(train_ds, 16, True)#, persistent_workers=True, num_workers=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(300):
        errs_train = []
        model = model.train()
        for _, batch, volume in tqdm.tqdm(train_loader):
            batch = batch.to(device)
            volume = volume.to(device)
            pred, trans, l = model(batch)
            ppred, _ = decoder(l)
            #loss = ((pred - volume) ** 2).mean()
            loss = pointnet_utils.feature_transform_reguliarzer(trans)
            loss = loss + utils_3d.chamfer_dist_slow_normal(ppred, batch, outlier_cutoff=1, normal_loss=False)
            errs_train.append(loss.item())
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:# and epoch > 0:
            utils_3d.visualize_point_clouds(batch[0, :, :3].detach().cpu().numpy(), ppred[0, :, :3].detach().cpu().numpy())

        #val_error = evaluate(model, val_loader)
        val_error = 0
        print(f"{epoch} Train: {np.mean(errs_train)}, Val: {val_error}")

    torch.save(model, out)

if __name__ == "__main__":
    train(Path(r"F:\Boxes-ds\segmented_distance_depth"), Path("split_without2024"), 
          Path(r"C:\Users\Admin\Desktop\master_thesis\volume_prediction_fip\3dvol_approx\local_stuff\vol_model.pth"))
    
