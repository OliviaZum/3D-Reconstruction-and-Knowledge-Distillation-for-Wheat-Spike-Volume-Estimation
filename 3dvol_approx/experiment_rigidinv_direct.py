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
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

"""
Results:
ply - direct (~100 epochs):
    - Corr: 0.98
    - MAE: 188
real - direct (~100 epochs)
    - Corr: 0.75
    - MAE: 605
"""

def evaluate(dataloader: DataLoader, model: nn.Module):
    model = model.eval()
    with torch.no_grad():
        generator = torch.Generator(device)
        generator.manual_seed(0)
        dataloader.dataset.reset_generator()
        vol_pred = []
        vol_real = []
        for _, batch, vol_batch in dataloader:
            batch = batch.to(device)
            r = utils_3d.to_rigid_invariant_representation(batch[:, :, 0:3], generator=None)
            volume = model(r)
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
        if True:
            plt.plot((2000, 8500), (2000, 8500))
            plt.scatter(datasets_3d.DepthMapDataset.vol_unorm(vol_pred), datasets_3d.DepthMapDataset.vol_unorm(vol_real))
            plt.show()
        return loss

if __name__ == "__main__":
    train_dataset, val_dataset = experiment_utils.get_real_dataset()
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = torch.load("3dvol_approx/local_stuff/ply2volume_model.pth", weights_only=False).to(device).eval()
    evaluate(val_loader, model)
    exit(0)

    model = models_3d.RigidInvariantPointNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15, 0.5)

    best_val = None
    best_model = None
    for epoch in range(51):
        errs_train = []
        model.train()
        for idx, batch, vol_batch in train_loader:
            batch = batch.to(device)
            vol_batch = vol_batch.to(device)
            r = utils_3d.to_rigid_invariant_representation(batch[:, :, 0:3])
            volume = model(r)
                
            loss = F.mse_loss(volume.squeeze(), vol_batch)
            errs_train.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        if epoch % 10 == 0 and epoch > 0:
            err_val = evaluate(val_loader, model)

            if best_val is None or err_val < best_val:
                best_val = err_val
                best_model = copy.deepcopy(model).cpu()

        print(f"epoch {epoch}. Train: {np.mean(errs_train)}")

    torch.save(best_model, "3dvol_approx/local_stuff/ply2volume_model.pth")