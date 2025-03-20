import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import copy
import experiments.shared.datasets_3d as datasets_3d
import experiments.shared.utils_3d as utils_3d
import experiments.shared.models_3d as models_3d
import experiment_utils
from torch.nn import functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
arti_2_vol = torch.load("3dvol_approx/local_stuff/ply2volume_model.pth", weights_only=False).to(device).eval()

"""
Deprecated.

Results:
    Train only incomplete -> complete (wasserstein) (500~)
    - Corr: 0.68, MAE: 790
    - 
    Train only incomplete -> complete (MSE) (500~)
    - Corr: 0.77, MAE: 610

    Train incomplete -> complete (MSE) Variational inference
    - Cor: 0.73, MAE: 655
"""

class RigidInvariantCompletion(nn.Module):
    def __init__(self, point_cloud_size_output = 1000, bins = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pnet = models_3d.RigidInvariantPointNet(output="latent", latent_size=256)
        self.last = nn.Sequential(
            nn.Linear(128, point_cloud_size_output * bins),
        )
        self.point_cloud_size_output = point_cloud_size_output
        self.bins = bins
        self.sample_eval = False
    
    def forward(self, x):
        latent = self.pnet(x)
        mu = latent[:, :128]
        logvar = latent[:, 128:]
        if self.training or self.sample_eval:
            x_samp = mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)
        else:
            x_samp = mu
        x = self.last(x_samp)
        x = x.reshape(-1, self.point_cloud_size_output, self.bins)
        #x = torch.nn.functional.softmax(x, dim=-1)
        if self.training:
            return x, latent
        else:
            return x
    
    def kldiv(self, latent, var_prior = 1):
        mu = latent[:, :128]
        logvar = latent[:, 128:]
        return 0.5 * torch.mean((logvar.exp() / var_prior) + (mu ** 2 / var_prior) - 1 - logvar)

def evaluate(dataloader: DataLoader, model: nn.Module, show_plot = False, vari_inf = False):
    model = model.eval()
    with torch.no_grad():
        vol_pred = []
        vol_real = []
        vars = []
        losses = []
        for _, batch_artificial, batch_real, vol_batch in dataloader:
            batch_real = batch_real.to(device)
            batch_artificial = batch_artificial.to(device)
            batch_artificial = utils_3d.to_rigid_invariant_representation(batch_artificial[:, :, 0:3], batch_artificial[:, :, 6])
            batch_real = utils_3d.to_rigid_invariant_representation(batch_real[:, :, 0:3], batch_real[:, :, 6])

            complete_pred = model(batch_real)
            loss = utils_3d.compare_rigid_invariant_mse(complete_pred, batch_artificial)
            losses.append(loss.cpu())
            
            if vari_inf:
                vps = []
                model.sample_eval = True
                for i in range(100):
                    complete_pred = model(batch_real)
                    vp = arti_2_vol(complete_pred).squeeze()
                    vps.append(vp)
                model.sample_eval = False
                vps = torch.stack(vps, dim=1)
                volume = vps.mean(dim=1).cpu()
                var = vps.var(dim=1).cpu()
                vars.append(var)
                vol_pred.append(volume)
            else:
                volume = arti_2_vol(complete_pred)
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
        if show_plot:
            plt.plot((2000, 8500), (2000, 8500))
            plt.scatter(datasets_3d.DepthMapDataset.vol_unorm(vol_real), datasets_3d.DepthMapDataset.vol_unorm(vol_pred))
            if len(vars) > 0:
                vars = torch.concat(vars).sqrt() * 1000
                plt.errorbar(datasets_3d.DepthMapDataset.vol_unorm(vol_real), datasets_3d.DepthMapDataset.vol_unorm(vol_pred), yerr=vars, fmt="o", capsize=3)
            plt.show()
        return loss

if __name__ == "__main__":
    train_dataset, val_dataset = experiment_utils.get_combined_dataset()
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = RigidInvariantCompletion().to(device)

    if True:
        model = torch.load("3dvol_approx/local_stuff/tmp_model.pth", weights_only=False).to(device)
        evaluate(val_loader, model, True, True)
        exit(0)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 40, 0.5)

    best_val = None
    best_model = None
    for epoch in range(221):
        errs_train = []
        model.train()
        for idx, batch_artificial, batch_real, vol_batch in train_loader:
            batch_real = batch_real.to(device)
            batch_artificial = batch_artificial.to(device)
            
            batch_real = utils_3d.to_rigid_invariant_representation(batch_real[:, :, 0:3], batch_real[:, :, 6])
            batch_artificial = utils_3d.to_rigid_invariant_representation(batch_artificial[:, :, 0:3], batch_artificial[:, :, 6])
            complete_pred, latent = model(batch_real)

            loss = utils_3d.compare_rigid_invariant_mse(complete_pred, batch_artificial)
            loss = loss + 0.01 * model.kldiv(latent)
            errs_train.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        if epoch % 20 == 0:# and epoch > 0:
            err_val = evaluate(val_loader, model)

            if best_val is None or err_val < best_val:
                best_val = err_val
                best_model = copy.deepcopy(model).cpu()

        print(f"epoch {epoch}. Train: {np.mean(errs_train)}")

    torch.save(best_model, "3dvol_approx/local_stuff/tmp_model.pth")