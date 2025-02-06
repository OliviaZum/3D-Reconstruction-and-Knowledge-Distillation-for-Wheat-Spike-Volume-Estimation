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
arti_2_vol = torch.load("3dvol_approx/local_stuff/ply2volume_model.pth", weights_only=False).to(device).eval()

"""
Results:
    Train only incomplete -> complete (wasserstein) (500~)
    - Corr: 0.68, MAE: 790
    - 
    Train only incomplete -> complete (MSE) (500~)
    - Corr: 0.68, MAE: 711
"""

def evaluate(dataloader: DataLoader, model: nn.Module):
    model = model.eval()
    with torch.no_grad():
        vol_pred = []
        vol_real = []
        losses = []
        for _, batch_artificial, batch_real, vol_batch  in dataloader:
            batch_real = batch_real.to(device)
            batch_artificial = batch_artificial.to(device)
            batch_real = utils_3d.to_rigid_invariant_representation(batch_real[:, :, 0:3])
            batch_artificial = utils_3d.to_rigid_invariant_representation(batch_artificial[:, :, 0:3])

            complete_pred = model(batch_real)
            loss = utils_3d.compare_rigid_invariant_wasserstein(complete_pred, batch_artificial)
            losses.append(loss.cpu())

            volume = arti_2_vol(complete_pred)
            #volume = torch.zeros_like(vol_batch)
            volume = volume.cpu().squeeze()
            vol_pred.append(volume)
            vol_real.append(vol_batch)
        vol_pred = torch.concat(vol_pred)
        vol_real = torch.concat(vol_real)
        print(f"Val MAE: {F.l1_loss(datasets_3d.DepthMapDataset.vol_unorm(vol_pred), datasets_3d.DepthMapDataset.vol_unorm(vol_real))}")
        print(f"Val Corr: {np.corrcoef(vol_real, vol_pred)[0, 1]}")
        loss = np.mean(losses)
        print(f"Val Loss (hist diff): {loss}")
        #plt.plot((2000, 8500), (2000, 8500))
        #plt.scatter(datasets_3d.DepthMapDataset.vol_unorm(vol_pred), datasets_3d.DepthMapDataset.vol_unorm(vol_real))
        #plt.show()
        return loss
    
def evaluate2(dataloader: DataLoader, model: nn.Module):
    model = model.eval()
    torch.random.manual_seed(0)
    with torch.no_grad():
        vol_pred = []
        vol_real = []
        losses = []
        for _, batch_artificial, batch_real, vol_batch  in dataloader:
            batch_real = batch_real.to(device)
            batch_artificial = batch_artificial.to(device)
            batch_real = utils_3d.to_rigid_invariant_representation(batch_real[:, :, 0:3])
            batch_artificial = utils_3d.to_rigid_invariant_representation(batch_artificial[:, :, 0:3])

            complete_pred = model(batch_real)
            loss = utils_3d.compare_rigid_invariant_wasserstein(complete_pred, batch_artificial)
            losses.append(loss.cpu())

            vols = []
            for i in range(40):
                if i % 10 == 0:
                    complete_pred = model(batch_real)
                modl = torch.randn_like(complete_pred) * 0.01 + complete_pred
                v = arti_2_vol(modl)
                vols.append(v)
            vols = torch.concat(vols, dim=1)
            vt = vols.T
            volume = vols.mean(dim=1)
            #volume = arti_2_vol(complete_pred)
            #volume = torch.zeros_like(vol_batch)
            volume = volume.cpu().squeeze()
            vol_pred.append(volume)
            vol_real.append(vol_batch)
        vol_pred = torch.concat(vol_pred)
        vol_real = torch.concat(vol_real)
        print(f"Val MAE: {F.l1_loss(datasets_3d.DepthMapDataset.vol_unorm(vol_pred), datasets_3d.DepthMapDataset.vol_unorm(vol_real))}")
        print(f"Val Corr: {np.corrcoef(vol_real, vol_pred)[0, 1]}")
        A = np.vstack([vol_real, np.ones(len(vol_real))]).T
        linregcoeff, _, _, _ = np.linalg.lstsq(A, vol_pred, rcond=None)
        print(f"Steepness: {linregcoeff[0]}")
        loss = np.mean(losses)
        print(f"Val Loss (hist diff): {loss}")
        return loss

if __name__ == "__main__":
    train_dataset, val_dataset = experiment_utils.get_combined_dataset()
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=3, persistent_workers=True, prefetch_factor=6)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    #model = models_3d.RigidInvariantPointNet(output="pointfeatures", latent_size=512).to(device)
    model = models_3d.RigidInvariantCompletion().to(device)

    #model = torch.load("3dvol_approx/local_stuff/tmp_model.pth", weights_only=False).to(device)
    #evaluate(val_loader, model)
    #exit(0)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, 0.5)

    

    best_val = None
    best_model = None
    for epoch in range(501):
        errs_train = []
        model.train()
        for idx, batch_artificial, batch_real, vol_batch in train_loader:
            batch_real = batch_real.to(device)
            batch_artificial = batch_artificial.to(device)
            
            batch_real = utils_3d.to_rigid_invariant_representation(batch_real[:, :, 0:3])
            batch_artificial = utils_3d.to_rigid_invariant_representation(batch_artificial[:, :, 0:3])
            complete_pred = model(batch_real)

            loss = utils_3d.compare_rigid_invariant_wasserstein(complete_pred, batch_artificial)
            errs_train.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        if epoch % 20 == 0 and epoch > 0:
            err_val = evaluate(val_loader, model)

            if best_val is None or err_val < best_val:
                best_val = err_val
                best_model = copy.deepcopy(model).cpu()

        print(f"epoch {epoch}. Train: {np.mean(errs_train)}")

    torch.save(best_model, "3dvol_approx/local_stuff/tmp_model.pth")