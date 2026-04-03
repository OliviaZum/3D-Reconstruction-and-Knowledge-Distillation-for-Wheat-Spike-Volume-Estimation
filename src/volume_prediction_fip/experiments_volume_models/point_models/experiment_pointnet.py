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

        eps = 1e-8
        vol_pred_un = datasets.vol_unorm(vol_pred)
        vol_real_un = datasets.vol_unorm(vol_real)
        mape = torch.mean(torch.abs((vol_real_un - vol_pred_un) / (vol_real_un + eps))) * 100
        print(f"Val MAPE: {mape}")

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

#without scaling and normalizing
#def preproc_batch(batch, model_name):
#    with torch.no_grad():
#        # Normalize, but keep scale (why 60? It's more or less the size of most spikes, also used in Rigid
#        # Invariant Pointnet as distance cutoff)
#        batch[:, :, 0:3] = batch[:, :, 0:3] / 60
#        if model_name == "pointnet-direct-real-sampled.pth" or model_name == "pointnet2-direct-real-sampled.pth":
#            batch = samplu(batch)
#        if model_name == "pointnet2-direct-real.pth":
#            r = batch[:, :, 0:3].permute(0, 2, 1)
#        else:
#            r = batch[:, :, 0:6].permute(0, 2, 1)
#        return r

def preproc_batch(batch, model_name):
    with torch.no_grad():
        # 1. Center each spike (remove translation)
        #coords = batch[:, :, 0:3]
        #centroid = coords.mean(dim=1, keepdim=True)
        #coords = coords - centroid

        # 2. Global scaling for numerical stability (preserves relative size)
        #coords = coords / 60.0
        #batch[:, :, 0:3] = coords
        batch[:, :, 0:3] = batch[:, :, 0:3] / 60

         # Scale per spike
        #scale = coords.norm(dim=2).max(dim=1)[0].view(-1, 1, 1)
        #coords = coords / (scale + 1e-8)

        # 3. Optional sampling (unchanged)
        if model_name == "pointnet-direct-real-sampled.pth" or \
           model_name == "pointnet2-direct-real-sampled.pth":
            batch = samplu(batch)

        # 4. Prepare tensor shape for model
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
    
    #model_name = "pointnet-direct-real.pth"
    model_name = "pointnet-direct-real-sampled.pth"
    #model_name = "pointnet2-direct-real.pth"
    #model_name = "pointnet2-direct-real-sampled.pth"

    if (model_name == "pointnet-direct-real.pth" or model_name == "pointnet2-direct-real.pth"
        or "pointnet-direct-real-sampled.pth" == model_name or "pointnet2-direct-real-sampled.pth" == model_name):
        train_dataset, val_dataset, test_dataset = utils_experiment.get_real_dataset3d(force_recompute=False)
    train_loader = DataLoader(train_dataset, batch_size=7, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    evaluation = True
    if evaluation:

        model = torch.load(helpers.get_exp_path() / f"{model_name}", weights_only=False).to(device).eval()

        plot_pca = "none"

        if plot_pca == "pointnet":
            all_features = []
            all_volumes = []

            with torch.no_grad():
                for _, batch, vol_batch, mask, _ in test_loader:

                    batch = batch.to(device)
                    r = preproc_batch(batch, model_name)

                    # -------- GLOBAL FEATURE --------
                    features, _, _ = model.feat(r)   # (B, 1024)

                    all_features.append(features.cpu())
                    all_volumes.append(vol_batch.cpu())

            features = torch.cat(all_features)
            volumes = torch.cat(all_volumes)

            print("Feature shape:", features.shape)

            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            lat2d = pca.fit_transform(features.numpy())

            vol_mm3 = datasets.vol_unorm(volumes).numpy()

            plt.figure(figsize=(6,5))
            sc = plt.scatter(
                lat2d[:, 0],
                lat2d[:, 1],
                c=vol_mm3,
                cmap="viridis",
                s=30
            )
            plt.colorbar(sc, label="True volume [mm³]")
            plt.xlabel("PC1")
            plt.ylabel("PC2")

            # SAME LIMITS AS OTHER MODELS
            plt.xlim(-200, 200)
            plt.ylim(-200, 200)

            save_path = helpers.get_exp_path() / f"pca_{model_name}.png"
            plt.savefig(save_path, dpi=300)
            print(f"Saved PCA plot to {save_path}")

            corr_pc1 = np.corrcoef(lat2d[:,0], volumes.numpy())[0,1]
            corr_pc2 = np.corrcoef(lat2d[:,1], volumes.numpy())[0,1]

            print("Correlation PC1 vs normalized volume:", corr_pc1)
            print("Correlation PC2 vs normalized volume:", corr_pc2)
        
        elif plot_pca == "pointnet2": 
                all_features = []
                all_volumes = []

                with torch.no_grad():
                    for _, batch, vol_batch, mask, _ in test_loader:

                        batch = batch.to(device)
                        r = preproc_batch(batch, model_name)

                        # -------------------------
                        # Forward manually through SA layers
                        # -------------------------
                        l1_xyz, l1_points = model.sa1(r[:, :3, :], None)
                        l2_xyz, l2_points = model.sa2(l1_xyz, l1_points)
                        _, l3_points = model.sa3(l2_xyz, l2_points)

                        # l3_points: (B, 1024, 1)
                        features = l3_points.view(l3_points.size(0), -1)

                        all_features.append(features.cpu())
                        all_volumes.append(vol_batch.cpu())

                features = torch.cat(all_features)
                volumes = torch.cat(all_volumes)

                print("Feature shape:", features.shape)

                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                lat2d = pca.fit_transform(features.numpy())

                vol_mm3 = datasets.vol_unorm(volumes).numpy()

                plt.figure(figsize=(6,5))
                sc = plt.scatter(
                    lat2d[:, 0],
                    lat2d[:, 1],
                    c=vol_mm3,
                    cmap="viridis",
                    s=30
                )
                plt.colorbar(sc, label="True volume [mm³]")
                plt.xlabel("PC1")
                plt.ylabel("PC2")

                plt.xlim(-75, 75)
                plt.ylim(-75, 75)

                save_path = helpers.get_exp_path() / f"pca_{model_name}.png"
                plt.savefig(save_path, dpi=300)
                print(f"Saved PCA plot to {save_path}")

                corr_pc1 = np.corrcoef(lat2d[:,0], volumes.numpy())[0,1]
                corr_pc2 = np.corrcoef(lat2d[:,1], volumes.numpy())[0,1]

                print("Correlation PC1 vs normalized volume:", corr_pc1)
                print("Correlation PC2 vs normalized volume:", corr_pc2)

        else: 
            evaluate(test_loader, model, model_name, show_plot=True)

        exit(0)

    if model_name == "pointnet2-direct-real.pth" or model_name == "pointnet2-direct-real-sampled.pth":
        model = pointnet_utils2.PointNet2ClsMsg(1, normal_channel=False).to(device)
        nepoch = 60 # This is healla slow and seems anyway to not converge much farther than a simple PointNet
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

#point net direct real: #
#Val MAE: 787.1990966796875
#Val Corr: 0.5674769042395481
#Val Loss: 1.051011562347412
#Steepness: 0.26921453120572586



#pointnet2
#Val MAE: 688.924560546875
#Val MAPE: 15.631975173950195
#Val Corr: 0.7120191691125239
#Val Loss: 0.7516993284225464
#Steepness: 0.519321129949181