"""
Uses the ply2volume_model (can be trained with experiment_rigidinv_direct) to train
a RigidInvariantPointNet for volume prediction while using the features predicted by the
ply2volume_model on the corresponding spike ply file as additional loss.
Outperforms direct volume prediction on real data slightly.

Performance on test:
Val MAE: 564.2942504882812
Val Corr: 0.8091378891713066
Steepness: 0.736374898259229
"""

import accelerate
import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import copy
from torch.nn import functional as F
import matplotlib.pyplot as plt

from volume_prediction_fip.experiments_volume_models.shared import utils_experiment, models, datasets, utils_3d
from volume_prediction_fip.utils import helpers

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
arti_2_vol = torch.load(helpers.get_exp_path() / "ply2volume_model.pth", weights_only=False).to(device).eval()
arti_2_vol.output = "features"

def evaluate(dataloader: DataLoader, model: nn.Module, latw = 5, show_plot = False):
    model = model.eval()
    with torch.no_grad():
        vol_pred = []
        vol_real = []
        lat_losses = []

        for _, data_ply, data_depthmap, plymask, mask, vol_batch, _ in dataloader:
            data_ply, data_depthmap, plymask, mask, vol_batch = [t.to(device) for t in (data_ply, data_depthmap, plymask, mask, vol_batch)]
            data_depthmap = utils_3d.to_rigid_invariant_representation(data_depthmap[:, :, 0:3], data_depthmap[:, :, 6])
            data_ply = utils_3d.to_rigid_invariant_representation(data_ply[:, :, 0:3], data_ply[:, :, 6])

            ply_latent = arti_2_vol(data_ply, plymask)
            volume, latent = model(data_depthmap, mask)
        
            lat_loss = ((ply_latent - latent) ** 2).mean(dim=1)
            lat_losses.append(lat_loss.cpu())

            vol_pred.append(volume.cpu().squeeze())
            vol_real.append(vol_batch.cpu().squeeze())

        vol_pred = torch.concat(vol_pred)
        vol_real = torch.concat(vol_real)
        lat_losses = torch.concat(lat_losses)

        volloss = ((vol_pred - vol_real) ** 2).mean().item()
        latloss = lat_losses.mean().item()
        loss = volloss + latloss * latw
        print(f"Val MAE: {F.l1_loss(datasets.vol_unorm(vol_pred), datasets.vol_unorm(vol_real))}")
        print(f"Val Corr: {np.corrcoef(vol_real, vol_pred)[0, 1]}")
        A = np.vstack([vol_real, np.ones(len(vol_real))]).T
        linregcoeff, _, _, _ = np.linalg.lstsq(A, vol_pred, rcond=None)
        print(f"Steepness: {linregcoeff[0]}")
        #print(f"Val Loss (+latent diff): {loss}")
        #print(f"Val Loss volume: {volloss}")
        #print(f"Val Loss latent: {latloss}")
        if show_plot:
            plt.plot((2000, 8500), (2000, 8500))
            plt.scatter(datasets.vol_unorm(vol_real), datasets.vol_unorm(vol_pred))
            plt.show()
        return loss

if __name__ == "__main__":
    accelerate.utils.set_seed(1, deterministic=True)
    train_dataset, val_dataset, test_dataset = utils_experiment.get_combined_point_real_arti_dataset()
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = models.RigidInvariantPointNet(output="volumelatent").to(device)

    model_name = "rigidinv_indirect2.pth"
    if True:
        model = torch.load(helpers.get_exp_path() / f"{model_name}", weights_only=False).to(device)
        evaluate(test_loader, model, 5, True)
        exit(0)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.5)
    latw=5

    best_val = None
    best_model = None
    for epoch in range(51):
        errs_train = []
        model.train()
        for _, data_ply, data_depthmap, plymask, mask, vol_batch, _ in train_loader:
            data_ply, data_depthmap, plymask, mask, vol_batch = [t.to(device) for t in (data_ply, data_depthmap, plymask, mask, vol_batch)]
            
            data_depthmap = utils_3d.to_rigid_invariant_representation(data_depthmap[:, :, 0:3], data_depthmap[:, :, 6])
            data_ply = utils_3d.to_rigid_invariant_representation(data_ply[:, :, 0:3], data_ply[:, :, 6])
            ply_latent = arti_2_vol(data_ply, plymask)

            volume, latent = model(data_depthmap, mask)
            v_loss = ((volume.squeeze() - vol_batch) ** 2).mean()
            lat_loss = ((ply_latent - latent) ** 2).mean()
            loss = v_loss + lat_loss * latw
            errs_train.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        err_val = evaluate(val_loader, model, latw=latw)

        #if best_val is None or err_val < best_val:
        #    best_val = err_val
        #    best_model = copy.deepcopy(model).cpu()

        # Not terribly clean, but training seems to stabilize towards the end resulting in higher test performance,
        #  but somewhat lower val performance
        best_model = copy.deepcopy(model).cpu()

        print(f"epoch {epoch}. Train: {np.mean(errs_train)}")
    torch.save(best_model, helpers.get_exp_path() / f"{model_name}")