import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import copy
import datasets_3d
import utils_3d
import models_3d
import experiment_utils
from torch.nn import functional as F

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

"""
Results: 
    - With Chamfer secondary loss (~100 epochs):
        - Corr: 0.59
        - MAE: 724
    - Without secondary loss, only mse (~100 epochs):
        - Corr:  0.56
        - MAE: 765
"""

def evaluate(dataloader: DataLoader, model: nn.Module):
    model = model.eval()
    with torch.no_grad():
        vol_pred = []
        vol_real = []
        for _, batch, vol_batch in dataloader:
            batch = batch.to(device)
            volume, _ = model(batch)
            volume = volume.cpu().squeeze()
            vol_pred.append(volume)
            vol_real.append(vol_batch)
        vol_pred = torch.concat(vol_pred)
        vol_real = torch.concat(vol_real)
        loss = F.mse_loss(vol_real, vol_pred)
        print(f"Val MAE: {F.l1_loss(datasets_3d.DepthMapDataset.vol_unorm(vol_pred), datasets_3d.DepthMapDataset.vol_unorm(vol_real))}")
        print(f"Val Corr: {np.corrcoef(vol_real, vol_pred)[0, 1]}")
        return loss

if __name__ == "__main__":
    train_dataset, val_dataset = experiment_utils.get_real_dataset()
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = models_3d.PointCloudAutoEncoder_volume().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, 0.5)

    best_val = None
    best_model = None
    for epoch in range(201):
        errs_train = []
        model.train()
        for idx, batch, vol_batch in train_loader:
            batch = batch.to(device)
            vol_batch = vol_batch.to(device)
            volume, (predictions, coarse_pred), _ = model(batch)
            #loss = utils_3d.chamfer_dist_slow_normal(predictions, batch, normal_loss=False)
            #loss = loss + utils_3d.chamfer_dist_slow_normal(coarse_pred, batch, normal_loss=False)
                
            loss = F.mse_loss(volume.squeeze(), vol_batch)
            errs_train.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        if epoch % 2 == 0 and epoch > 0:
            err_val = evaluate(val_loader, model)

            if best_val is None or err_val < best_val:
                best_val = err_val
                best_model = copy.deepcopy(model).cpu()

        print(f"epoch {epoch}. Train: {np.mean(errs_train)}")