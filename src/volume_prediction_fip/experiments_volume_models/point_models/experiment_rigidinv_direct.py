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
Val MAE: 141.0824432373047
Val Corr: 0.9881163633508186
Steepness: 0.9930740456518544

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
from torch.nn import functional as F
import matplotlib.pyplot as plt
import accelerate
from volume_prediction_fip.experiments_volume_models.shared import utils_experiment, models, datasets, utils_3d
from volume_prediction_fip.utils import helpers
import time



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def save_latents(dataloader: DataLoader, model: nn.Module, bins: int, out_name: str):
    model = model.eval()
    model.output = "feature"   # IMPORTANT

    latents = {}

    with torch.no_grad():
        for idx, batch, _, mask, _ in dataloader:
            batch = batch.to(device)
            mask = mask.to(device)

            r = utils_3d.to_rigid_invariant_representation(
                batch[:, :, 0:3],
                batch[:, :, 6],
                num_samples=None,
            )

            latent = model(r, mask)  # [B, latent_dim]

            for i, sample_idx in enumerate(idx):
                latents[int(sample_idx)] = latent[i].cpu()

    out_path = helpers.get_exp_path() / out_name
    torch.save(latents, out_path)
    print(f"Saved {len(latents)} latents to {out_path}")
        

def evaluate(dataloader: DataLoader, model: nn.Module, show_plot = False, bins=10):
    model = model.eval()
    with torch.no_grad():
        vol_pred = []
        vol_real = []
        weights = []
        for _, batch, vol_batch, mask, weight in dataloader:
            batch = batch.to(device)
            mask = mask.to(device)
            r = utils_3d.to_rigid_invariant_representation(batch[:, :, 0:3], batch[:, :, 6], num_samples=None, bins=bins)
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
        vp = datasets.vol_unorm(vol_pred)
        vr = datasets.vol_unorm(vol_real)
        print(f"Val MAPE: {(torch.abs((vr - vp) / (vr + 1e-8))).mean().item() * 100}")
        if show_plot:
            plt.plot((2000, 8500), (2000, 8500))
            plt.scatter(datasets.vol_unorm(vol_real), datasets.vol_unorm(vol_pred))
            plt.show()
        return loss

if __name__ == "__main__":
    # Results vary quite a bit depending on seed
    accelerate.utils.set_seed(1, deterministic=True)
    
    #model_name = "real_volume_model-direct.pth"
    model_name = "ply2volume_model_new.pth"
    #model_name = "ply2volume_voxelized_model.pth"
    #model_name = "fiplike_uniform_ply_model.pth"
    #model_name = "fiplike_normal_ply_model.pth"

    if model_name == "ply2volume_model_new.pth":
        train_dataset, val_dataset, test_dataset = utils_experiment.get_ply_dataset(force_recompute=False, voxelize=True)
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

    ##########################
    bins = 30
    print(f'Number of Bins: {bins}')
    evaluation = True
    ##########################


    if evaluation == True:
        model = torch.load(helpers.get_exp_path() / f"{model_name}", weights_only=False).to(device).eval()

         # ---- evaluation timing ----
        torch.cuda.synchronize()
        eval_start = time.perf_counter()

        plot_pca = True
        if plot_pca == True: 
                    
            # IMPORTANT: switch model to latent mode
            model.output = "latent"

            all_features = []
            all_volumes = []

            with torch.no_grad():
                for _, batch, vol_batch, mask, _ in test_loader:
                    batch = batch.to(device)
                    mask = mask.to(device)

                    r = utils_3d.to_rigid_invariant_representation(
                        batch[:, :, 0:3],
                        batch[:, :, 6],
                        num_samples=None,
                        bins=bins
                    )

                    latent = model(r, mask)  # (B, latent_dim)
                    all_features.append(latent.cpu())
                    all_volumes.append(vol_batch.cpu())

            features = torch.cat(all_features)
            volumes = torch.cat(all_volumes)

            print("Feature shape:", features.shape)
            print("Volume shape:", volumes.shape)

            features_np = features.numpy()
            volumes_np = volumes.numpy()

            from sklearn.decomposition import PCA

            pca = PCA(n_components=2)
            lat2d = pca.fit_transform(features_np)

             # ---- explained variance ----
            expl_var = pca.explained_variance_ratio_ * 100
            print(f"Explained variance PC1: {expl_var[0]:.2f}%")
            print(f"Explained variance PC2: {expl_var[1]:.2f}%")
            print(f"Cumulative (2 PCs): {(expl_var[0] + expl_var[1]):.2f}%")

            vol_mm3 = datasets.vol_unorm(torch.tensor(volumes_np)).numpy()

            plt.figure(figsize=(6,5))
            sc = plt.scatter(
                lat2d[:, 0],
                lat2d[:, 1],
                c=vol_mm3,
                cmap="viridis",
                s=30
            )
            cbar = plt.colorbar(sc)
            cbar.set_label("True volume [mm³]", fontsize=20)
            cbar.ax.tick_params(labelsize=14)
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.xlabel(f"PC1 ({expl_var[0]:.1f}%)", fontsize=22)
            plt.ylabel(f"PC2 ({expl_var[1]:.1f}%)", fontsize=22)

            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)


            # use SAME axis limit as ensemble
            plt.xlim(-22, 22)
            plt.ylim(-22, 22)

            save_path = helpers.get_exp_path() / f"pca_rigidinv_direct_{model_name}.png"
            plt.tight_layout()
            plt.savefig(save_path, dpi=600)
            print(f"Saved PCA plot to {save_path}")

            corr_pc1 = np.corrcoef(lat2d[:,0], volumes_np)[0,1]
            corr_pc2 = np.corrcoef(lat2d[:,1], volumes_np)[0,1]
            print("Correlation PC1 vs normalized volume:", corr_pc1)
            print("Correlation PC2 vs normalized volume:", corr_pc2)


        else:
            evaluate(test_loader, model, show_plot=True, bins=bins)

        torch.cuda.synchronize()
        eval_time = time.perf_counter() - eval_start

        print(f"Evaluation time: {eval_time:.2f}s")

        #save_latents(
        #    dataloader=DataLoader(train_dataset, batch_size=8, shuffle=False),
        #    model=model,
        #    out_name=f"{model_name.replace('.pth','')}_bins{bins}_train_latents.pt"
        #    bins=bins,
        #)

        #save_latents(
        #    dataloader=DataLoader(val_dataset, batch_size=8, shuffle=False),
        #    model=model,
        #    bins=bins,
        #    out_name=f"{model_name.replace('.pth','')}_bins{bins}_val_latents.pt"
        #)

        #save_latents(
        #    dataloader=DataLoader(test_dataset, batch_size=8, shuffle=False),
        #    model=model,
        #    bins=bins,
        #    out_name=f"{model_name.replace('.pth','')}_bins{bins}_test_latents.pt"
        #)

        exit(0)

    model = models.RigidInvariantPointNet(bins=bins).to(device)

    """
    Using a pretrained model (does not work better than just direct training; deprecated)

    pretrain_model = torch.load(helpers.get_exp_path() / "pretrained_model.pth", weights_only=False)
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
        torch.cuda.synchronize()
        epoch_start = time.perf_counter()

        errs_train = []
        model.train()

        for idx, batch, vol_batch, mask, weight in train_loader:
            vol_batch = vol_batch.to(device)
            mask = mask.to(device)
            batch = batch.to(device)
            weight = weight.to(device)
            
            r = utils_3d.to_rigid_invariant_representation(batch[:, :, 0:3], batch[:, :, 6], num_samples=None, bins = bins)
            volume = model(r, mask)
                
            loss = ((volume.squeeze() - vol_batch) ** 2).mean()
            errs_train.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        err_val = evaluate(val_loader, model, bins=bins)

        if best_val is None or err_val < best_val:
            best_val = err_val
            best_model = copy.deepcopy(model).cpu()
        
        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start

        print(
            f"epoch {epoch:03d} | "
            f"train loss: {np.mean(errs_train):.4f} | "
            f"time: {epoch_time:.2f}s"
        )

        print(f"epoch {epoch}. Train: {np.mean(errs_train)}")

    torch.save(best_model, helpers.get_exp_path() / f"{model_name}")

#final

#learning rate: 
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.5)
#take best validation model 




#real dataset: otpitimzer 10, 0.5

#Number of Bins: 5
#Val MAE: 628.263916015625
#Val Corr: 0.7807985888032245
#Val Loss: 0.6358693838119507
#Steepness: 0.5345923046582944
#Val MAPE: 14.111658930778503
#Evaluation time: 0.37s

#Number of Bins: 10
#Val MAE: 577.6362915039062
#Val Corr: 0.8116039212084282
#Val Loss: 0.5376783013343811
#Steepness: 0.7108362061098719
#Val MAPE: 13.212825357913971
#Evaluation time: 0.42s

#Number of bins 15
#Val MAE: 567.2279052734375
#Val Corr: 0.8152441481652233
#Val Loss: 0.5269496440887451
#Steepness: 0.6307004556717791
#Evaluation time: 0.43s
#Val MAPE: 13.018052279949188

    
#Number of Bins: 20
#Val MAE: 569.1702880859375
#Val Loss: 0.5104888677597046
#Val Corr: 0.8213859839749641
#Steepness: 0.6769545099492628
#Val MAPE: 12.915945053100586
#Evaluation time: 0.48s
                
#bins 10: optimizer 5, 0.2
#Number of Bins: 10
#Val MAE: 584.6538696289062
#Val Corr: 0.8081667876255458
#Val Loss: 0.541265606880188
#Steepness: 0.6708369714019186
#Val MAPE: 13.3138969540596
#Evaluation time: 0.40s

#bins10, optimizer 2, 0.2
#Number of Bins: 10
#Val MAE: 606.1171875
#Val Corr: 0.7915556022101682
#Val Loss: 0.5837835669517517
#Steepness: 0.622086008428909
#Val MAPE: 13.704714179039001
#Evaluation time: 0.41s