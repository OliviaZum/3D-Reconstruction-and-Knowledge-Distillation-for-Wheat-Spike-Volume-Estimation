"""
Directly training the RigidInvariantPointNet for volume prediction.
By changing the modelname different datasets can be used (the real dataset,
the ply dataset [i.e. points sampled from the 3d scans], or the fiplike artificial
dataset)

Performance on real:
Val MAE: 580.3414916992188
Val Corr: 0.8002600909545133
Steepness: 0.6788825071122576

Performance on ply files:
Val MAE: 136.79244995117188
Val Corr: 0.989288331764323
Steepness: 1.0001779871566243

Performance on ply files (voxelized):
Val MAE: 108.59272766113281
Val Corr: 0.9923584956654167
Steepness: 1.00016117094416

Fiplike uniform:
Val MAE: 333.9336242675781
Val Corr: 0.9393371638387208
Steepness: 0.9940054408544241

Fiplike normal:
Val MAE: 284.6448059082031
Val Corr: 0.9530460571009036
Steepness: 0.9279888669048981
"""

import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import copy
import experiments_volume_models.shared.datasets as datasets
import experiments_volume_models.shared.utils_3d as utils_3d
import experiments_volume_models.shared.models as models
import experiments_volume_models.shared.utils_experiment as utils_experiment
from torch.nn import functional as F
import matplotlib.pyplot as plt
import accelerate

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def evaluate(dataloader: DataLoader, model: nn.Module, show_plot = False):
    model = model.eval()
    with torch.no_grad():
        vol_pred = []
        vol_real = []
        weights = []
        for _, batch, vol_batch, mask, weight in dataloader:
            batch = batch.to(device)
            mask = mask.to(device)
            r = utils_3d.to_rigid_invariant_representation(batch[:, :, 0:3], batch[:, :, 6], num_samples=None)
            volume = model(r, mask)
            volume = volume.cpu().squeeze()
            vol_pred.append(volume)
            vol_real.append(vol_batch)
            weights.append(weight)
        vol_pred = torch.concat(vol_pred)
        vol_real = torch.concat(vol_real)
        weights = torch.concat(weights)

        loss = ((vol_pred - vol_real) ** 2).mean().item()
        #loss = ((vol_pred - vol_real) ** 2).mean().item()
        print(f"Val MAE: {F.l1_loss(datasets.vol_unorm(vol_pred), datasets.vol_unorm(vol_real))}")
        print(f"Val Corr: {np.corrcoef(vol_real, vol_pred)[0, 1]}")
        print(f"Val Loss: {loss}")
        A = np.vstack([vol_real, np.ones(len(vol_real))]).T
        linregcoeff, _, _, _ = np.linalg.lstsq(A, vol_pred, rcond=None)
        print(f"Steepness: {linregcoeff[0]}")
        if show_plot:
            plt.plot((2000, 8500), (2000, 8500))
            plt.scatter(datasets.vol_unorm(vol_real), datasets.vol_unorm(vol_pred))
            plt.show()
        return loss

if __name__ == "__main__":
    # Results vary quite a bit depending on seed
    accelerate.utils.set_seed(1, deterministic=True)
    
    #model_name = "real_volume_model-direct.pth"
    #model_name = "ply2volume_model.pth"
    #model_name = "ply2volume_voxelized_model.pth"
    #model_name = "fiplike_uniform_ply_model.pth"
    model_name = "fiplike_normal_ply_model.pth"

    if model_name == "ply2volume_model.pth":
        train_dataset, val_dataset, test_dataset = utils_experiment.get_ply_dataset(force_recompute=False, voxelize=False)
    elif model_name == "ply2volume_voxelized_model.pth":
        train_dataset, val_dataset, test_dataset = utils_experiment.get_ply_dataset(force_recompute=False, voxelize=True)
    elif model_name == "real_volume_model-direct.pth":
        train_dataset, val_dataset, test_dataset = utils_experiment.get_real_dataset3d(force_recompute=False)
    elif model_name == "fiplike_uniform_ply_model.pth":
        train_dataset, val_dataset, test_dataset = utils_experiment.get_artificial_dataset_fiplike_uniform("depth", force_recompute=False)
    elif model_name == "fiplike_normal_ply_model.pth":
        train_dataset, val_dataset, test_dataset = utils_experiment.get_artificial_dataset_fiplike_normal("depth", force_recompute=False)
    else:
        assert False
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


    if True:
        model = torch.load(f"experiments_volume_models/local_stuff/{model_name}", weights_only=False).to(device).eval()
        evaluate(test_loader, model, show_plot=True)
        exit(0)

    model = models.RigidInvariantPointNet(bins=10).to(device)

    """
    Using a pretrained model (does not work better than just direct training; deprecated)

    pretrain_model = torch.load("experiments_volume_models/local_stuff/pretrained_model.pth", weights_only=False)
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.5)

    best_val = None
    best_model = None
    for epoch in range(60):
        errs_train = []
        model.train()
        for idx, batch, vol_batch, mask, weight in train_loader:
            vol_batch = vol_batch.to(device)
            mask = mask.to(device)
            batch = batch.to(device)
            weight = weight.to(device)
            
            r = utils_3d.to_rigid_invariant_representation(batch[:, :, 0:3], batch[:, :, 6], num_samples=None)
            volume = model(r, mask)
                
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

    torch.save(best_model, f"experiments_volume_models/local_stuff/{model_name}")