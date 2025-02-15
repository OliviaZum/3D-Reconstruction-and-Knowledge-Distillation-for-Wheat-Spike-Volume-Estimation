import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import copy
import datasets_3d
import utils_3d
import models_3d
import utils_experiment
from torch.nn import functional as F
import matplotlib.pyplot as plt
import pandas as pd

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

"""
Results:
ply - direct (~100 epochs):
    - Corr: 0.98
    - MAE: 188
real - direct (~100 epochs)
    - Corr: 0.81
    - MAE: 546
    - Steepness: 0.75
real - direct, scaled (~100 epochs)
    - Corr: 0.76
    - MAE: 673
    - Steepness: 0.82
"""

def evaluate(dataloader: DataLoader, model: nn.Module, show_plot = False):
    model = model.eval()
    with torch.no_grad():
        vol_pred = []
        vol_real = []
        for _, batch, vol_batch, mask in dataloader:
            batch = batch.to(device)
            mask = mask.to(device)
            r = utils_3d.to_rigid_invariant_representation(batch[:, :, 0:3], batch[:, :, 6], num_samples=None)
            volume, _ = model(r, mask)
            volume = volume.cpu().squeeze()
            vol_pred.append(volume)
            vol_real.append(vol_batch)
        vol_pred = torch.concat(vol_pred)
        vol_real = torch.concat(vol_real)

        loss = F.mse_loss(vol_pred, vol_real)
        print(f"Val MAE: {F.l1_loss(datasets_3d.DepthMapDataset.vol_unorm(vol_pred), datasets_3d.DepthMapDataset.vol_unorm(vol_real))}")
        print(f"Val Corr: {np.corrcoef(vol_real, vol_pred)[0, 1]}")
        print(f"Val Loss: {loss}")
        A = np.vstack([vol_real, np.ones(len(vol_real))]).T
        linregcoeff, _, _, _ = np.linalg.lstsq(A, vol_pred, rcond=None)
        print(f"Steepness: {linregcoeff[0]}")
        if show_plot:
            plt.plot((2000, 8500), (2000, 8500))
            plt.scatter(datasets_3d.DepthMapDataset.vol_unorm(vol_real), datasets_3d.DepthMapDataset.vol_unorm(vol_pred))
            plt.show()
        return loss

if __name__ == "__main__":
    model_name = "real_volume_model.pth" # real_volume
    if model_name == "ply2volume_model.pth":
        train_dataset, val_dataset = utils_experiment.get_ply_dataset()
    elif model_name == "real_volume_model.pth":
        train_dataset, val_dataset = utils_experiment.get_real_dataset(force_recompute=False)
    else:
        assert False
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


    if False:
        model = torch.load(f"3dvol_approx/local_stuff/{model_name}", weights_only=False).to(device).eval()
        evaluate(val_loader, model, show_plot=True)
        exit(0)

    model = models_3d.RigidInvariantPointNet(bins=10).to(device)
    #model = models_3d.RigidInvariantTr().to(device)

    """
    Using a pretrained model

    pretrain_model = torch.load("3dvol_approx/local_stuff/pretrained_model.pth", weights_only=False)
    new_state_dict = model.state_dict()
    for k, v in pretrain_model.state_dict().items():
        if "l1" in k:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        [{"params": model.l1.parameters(), "lr": 0.0001},
         {"params": model.lastlin.parameters(), "lr": 0.002}], lr=0.002)
    """
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)

    best_val = None
    best_model = None
    for epoch in range(40):
        errs_train = []
        model.train()
        for idx, batch, vol_batch, mask in train_loader:
            vol_batch = vol_batch.to(device)
            mask = mask.to(device)
            batch = batch.to(device)
            #for i in range(len(batch)):
            #    utils_3d.visualize_point_clouds(batch[i, :, 0:3].cpu().detach().numpy())
            r = utils_3d.to_rigid_invariant_representation(batch[:, :, 0:3], batch[:, :, 6], num_samples=None)
            volume, _ = model(r, mask)
                
            #loss = ((volume.squeeze() - vol_batch) ** 2 * scale).mean()
            loss = ((volume.squeeze() - vol_batch) ** 2).mean()
            errs_train.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        err_val = evaluate(val_loader, model)

        if best_val is None or err_val < best_val:
            best_val = err_val
            best_model = copy.deepcopy(model).cpu()

        print(f"epoch {epoch}. Train: {np.mean(errs_train)}")

    torch.save(model, f"3dvol_approx/local_stuff/{model_name}")