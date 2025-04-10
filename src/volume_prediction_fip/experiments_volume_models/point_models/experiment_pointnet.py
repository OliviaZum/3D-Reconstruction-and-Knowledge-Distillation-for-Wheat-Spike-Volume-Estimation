import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import copy
from torch.nn import functional as F
import matplotlib.pyplot as plt
import accelerate

from volume_prediction_fip.experiments_volume_models.shared import utils_experiment, datasets
from volume_prediction_fip.experiments_volume_models.point_models import pointnet_utils, pointnet_utils2
from volume_prediction_fip.utils import helpers

"""
PointNet on test:
Val MAE: 802.4710693359375
Val Corr: 0.4789765149589325
Steepness: 0.22554833398026705

PointNet2 on test:
Val MAE: 744.63671875
Val Corr: 0.6252473159193865
Steepness: 0.3981724023623164

Sampled works worse
"""

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def evaluate(dataloader: DataLoader, model: nn.Module, model_name, show_plot = False):
    model = model.eval()
    with torch.no_grad():
        vol_pred = []
        vol_real = []
        weights = []
        for _, batch, vol_batch, mask, weight in dataloader:
            batch = batch.to(device)
            mask = mask.to(device)
            r = preproc_batch(batch, model_name)
            volume, _ = model(r)
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

def samplu(batch):
    n = 1000
    rt = torch.zeros((batch.shape[0], n, 6), device=batch.device)

    torch.use_deterministic_algorithms(False)
    for i in range(batch.shape[0]):
        w = batch[i, :, 6]
        w = w / w.sum()
        indices = torch.multinomial(w, num_samples=n, replacement=True)

        r = batch[i, indices, 0:6]
        rt[i] = r
    torch.use_deterministic_algorithms(True)

    return rt

def preproc_batch(batch, model_name):
    with torch.no_grad():
        # Normalize, but keep scale (why 60? It's more or less the size of most spikes, also used in Rigid
        # Invariant Pointnet as distance cutoff)
        batch[:, :, 0:3] / 60
        if model_name == "pointnet-direct-real-sampled.pth" or model_name == "pointnet2-direct-real-sampled.pth":
            batch = samplu(batch)
        if model_name == "pointnet2-direct-real.pth":
            r = batch[:, :, 0:3].permute(0, 2, 1)
        else:
            r = batch[:, :, 0:6].permute(0, 2, 1)
        return r


# The implementation does not support masks or weights, or batch size of 1, hence pointclouds are padded with 0 points
# to have equal size and weights are not used. (Yeah this is not nice - from prior experiments it is clear
# that the results are quite horrible anyway, weights and masks or not. So not investing a large amount of time in making this proper)

# Loss scaling is not there since it decreases correlation for those models also

if __name__ == "__main__":
    # Results vary quite a bit depending on seed
    accelerate.utils.set_seed(1, deterministic=True)
    
    model_name = "pointnet-direct-real.pth"
    #model_name = "pointnet-direct-real-sampled.pth"
    #model_name = "pointnet2-direct-real.pth"
    #model_name = "pointnet2-direct-real-sampled.pth"

    if (model_name == "pointnet-direct-real.pth" or model_name == "pointnet2-direct-real.pth"
        or "pointnet-direct-real-sampled.pth" == model_name or "pointnet2-direct-real-sampled.pth" == model_name):
        train_dataset, val_dataset, test_dataset = utils_experiment.get_real_dataset3d(force_recompute=False)
    train_loader = DataLoader(train_dataset, batch_size=7, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


    if False:
        model = torch.load(helpers.get_exp_path() / f"{model_name}", weights_only=False).to(device).eval()
        evaluate(test_loader, model, model_name, show_plot=True)
        exit(0)

    if model_name == "pointnet2-direct-real.pth" or model_name == "pointnet2-direct-real-sampled.pth":
        model = pointnet_utils2.PointNet2ClsMsg(1, normal_channel=False).to(device)
        nepoch = 15 # This is healla slow and seems anyway to not converge much farther than a simple PointNet
        # (Expected btw PointNet2 solves the problem of local features, which is not that relevant to this problem)
    elif model_name == "pointnet-direct-real.pth" or model_name == "pointnet-direct-real-sampled.pth":
        model = pointnet_utils.PointNetCls(k=1, normal_channel=True).to(device) # Using or not using normals does not really change performance
        tflossfn = pointnet_utils.PointNetClsLoss()
        nepoch = 60

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.5)

    best_val = None
    best_model = None
    for epoch in range(nepoch):
        errs_train = []
        model.train()
        for idx, batch, vol_batch, mask, weight in train_loader:
            vol_batch = vol_batch.to(device)
            mask = mask.to(device)
            batch = batch.to(device)
            weight = weight.to(device)
            
            r = preproc_batch(batch, model_name)
            volume, trans_feat = model(r)
            
            loss = ((volume.squeeze() - vol_batch) ** 2).mean() # Loss scale seems to decrease performance here, same as with RigidInvariant
            if model_name == "pointnet-direct-real.pth" or model_name == "pointnet-direct-real-sampled.pth":
                loss = loss + tflossfn(trans_feat)
            errs_train.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        err_val = evaluate(val_loader, model, model_name)

        if best_val is None or err_val < best_val:
            best_val = err_val
            best_model = copy.deepcopy(model).cpu()

        print(f"epoch {epoch}. Train: {np.mean(errs_train)}")

    torch.save(best_model, helpers.get_exp_path() / f"{model_name}")