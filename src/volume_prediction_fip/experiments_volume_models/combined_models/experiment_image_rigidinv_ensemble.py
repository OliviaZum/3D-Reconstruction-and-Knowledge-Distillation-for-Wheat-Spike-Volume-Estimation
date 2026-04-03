from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import copy
from torch.nn import functional as F
import matplotlib.pyplot as plt
import torch
import itertools
import accelerate

from volume_prediction_fip.experiments_volume_models.shared import utils_experiment, models, datasets, utils_3d
from volume_prediction_fip.utils import helpers
import time


"""
An ensemble of the RigidInvariant PointNet (trained with Scan supervision) and the regulated Transformer

Results on test (both models are trained directly):
Val MAE: 521.1376342773438
Val Corr: 0.8488245778778427
Steepness: 0.789510382404692

Results on test (regulated transformer is self distilled):
Val MAE: 531.038330078125
Val Corr: 0.8497957443458352
Steepness: 0.8254965274223702
"""

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def extract_ensemble_features(dataloader, model):
    model.eval()
    all_features = []
    all_volumes = []

    with torch.no_grad():
        for images, points, imagemask, pointmask, label, weight in dataloader:
            images = images.to(device)
            points = points.to(device)
            imagemask = imagemask.to(device)
            pointmask = pointmask.to(device)

            points = utils_3d.to_rigid_invariant_representation(
                points[:, :, 0:3],
                points[:, :, 6],
                num_samples=None
            )

            imgfeat = model.img_net(images, imagemask)
            imgfeat = model.prec_img(imgfeat)
            pointfeat = model.point_net(points, pointmask)

            combined = torch.cat((imgfeat, pointfeat), dim=-1)

            # 🔥 TRUE final spike-level embedding
            fused = model.final[:-1](combined)

            all_features.append(fused.cpu())
            all_volumes.append(label.cpu())

    return torch.cat(all_features), torch.cat(all_volumes)

def evaluate(dataloader: DataLoader, model: nn.Module, show_plot = False, save_test_predictions = None):
    model = model.eval()
    total_time = 0.0
    total_spikes = 0
    with torch.no_grad():
        vol_pred = []
        vol_real = []
        weights = []
        plants = []
        for images, points, imagemask, pointmask, label, weight in dataloader:
            images, points, imagemask, pointmask = [t.to(device) for t in [images, points, imagemask, pointmask]]
            points = utils_3d.to_rigid_invariant_representation(points[:, :, 0:3], points[:, :, 6], num_samples=None)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()

            volume = model(images, imagemask, points, pointmask)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end = time.perf_counter()

            total_time += (end - start)
            total_spikes += images.size(0)

            volume = volume.cpu().squeeze()
            vol_pred.append(volume)
            vol_real.append(label)
            weights.append(weight)
            #plants.extend(plant)
        print(f"Total inference time: {total_time:.4f}s")
        print(f"Inference time per spike: {total_time / total_spikes:.6f}s")
        vol_pred = torch.concat(vol_pred)
        vol_real = torch.concat(vol_real)
        weights = torch.concat(weights)


        if save_test_predictions is not None:
            import pandas as pd

            df = pd.DataFrame({
                #"plant_id": plants,
                "volume_pred_norm": vol_pred.numpy(),
                "volume_real_norm": vol_real.numpy(),
                "volume_pred_mm3": datasets.vol_unorm(vol_pred).numpy(),
                "volume_real_mm3": datasets.vol_unorm(vol_real).numpy(),
                "weight": weights.numpy()
            })

            df.to_csv(save_test_predictions, index=False)
            print(f"Saved predictions to {save_test_predictions}")


        mae = F.l1_loss(datasets.vol_unorm(vol_pred), datasets.vol_unorm(vol_real))
        print(f"Val MAE: {mae}")
        corr = np.corrcoef(vol_real, vol_pred)[0, 1]
        print(f"Val Corr: {corr}")
        print(f"Val Loss: {((vol_pred - vol_real) ** 2 * weights).mean().item()}")
        A = np.vstack([vol_real, np.ones(len(vol_real))]).T
        linregcoeff, _, _, _ = np.linalg.lstsq(A, vol_pred, rcond=None)
        print(f"Steepness: {linregcoeff[0]}")
        vp = datasets.vol_unorm(vol_pred)
        vr = datasets.vol_unorm(vol_real)
        print(f"Val MAPE: {(torch.abs((vr - vp) / (vr + 1e-8))).mean().item() * 100}")

        rate = -(- mae + corr * (50 / 0.03) - abs(1 - linregcoeff[0]) * (50 / 0.1))
        print(f"Rate: {rate}")
        if show_plot:
            plt.plot((2000, 8500), (2000, 8500))
            plt.scatter(datasets.vol_unorm(vol_real), datasets.vol_unorm(vol_pred))
            plt.show()
        
        return rate

if __name__ == "__main__":
    accelerate.utils.set_seed(0)

    train_dataset, val_dataset, test_dataset = utils_experiment.get_combined_image_point_dataset()

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    batch = next(iter(test_loader))
    print("Batch length:", len(batch))

    model_name = "3dglobimgensemble_new.pth"

    evaluation = True
    if evaluation == True:

        model = torch.load(helpers.get_exp_path() / f"{model_name}", weights_only=False).to(device).eval()
        plot_pca = True
        if plot_pca: 
            # ---- extract ensemble features ----
            features, volumes = extract_ensemble_features(test_loader, model)

            print("Feature shape:", features.shape)
            print("Volume shape:", volumes.shape)

            # convert to numpy
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


            plt.xlim(-22, 22)
            plt.ylim(-22, 22)
            plt.tight_layout()

            save_path = helpers.get_exp_path() / f"pca_ensemble_{model_name}.png"
            plt.savefig(save_path, dpi=600)
            print(f"Saved PCA plot to {save_path}")

            corr_pc1 = np.corrcoef(lat2d[:,0], volumes_np)[0,1]
            print("Correlation PC1 vs normalized volume:", corr_pc1)

        else: 

            evaluate(test_loader, model, show_plot=True, save_test_predictions=helpers.get_exp_path() / "test_predictions_ensemble.csv")
        exit(0)

    model = models.Image3dEnsemble(use_kd=False)

    pretrained_point = torch.load(helpers.get_exp_path() / f"rigidinv_indirect2_new.pth", weights_only=False)
    pretrained_img = torch.load(helpers.get_exp_path() / f"regulatedtransformer-direct.pth", weights_only=False)
    #pretrained_img = torch.load(helpers.get_exp_path() / f"self-distill-regulated_transformer.pth", weights_only=False)[]

    model.img_net.load_state_dict(pretrained_img.state_dict(), strict=False)
    #model.img_net.load_state_dict(pretrained_img, strict=False)
    model.point_net.load_state_dict(pretrained_point.state_dict(), strict=False)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(itertools.chain(model.final.parameters(), model.prec_img.parameters()), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)

    best_val = None
    best_model = None
    for epoch in range(30):
        torch.cuda.synchronize()
        epoch_start = time.perf_counter()

        errs_train = []
        model.train()
        for images, points, imagemask, pointmask, label, weight in train_loader:
            images, points, imagemask, pointmask, label, weight = [t.to(device) for t in [images, points, imagemask, pointmask, label, weight]]

            points = utils_3d.to_rigid_invariant_representation(points[:, :, 0:3], points[:, :, 6], num_samples=None, bins=10)
            volume = model(images, imagemask, points, pointmask)
                
            loss = ((volume.squeeze() - label) ** 2 * weight).mean()
            errs_train.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start

        n_train_spikes = len(train_dataset)

        print(
            f"epoch {epoch:03d} | "
            f"train loss: {np.mean(errs_train):.4f} | "
            f"time: {epoch_time:.2f}s | "
            f"time per spike: {epoch_time / n_train_spikes:.6f}s"
        )
        
        err_val = evaluate(val_loader, model)

        if best_val is None or (err_val < best_val and epoch > 15):
            best_val = err_val
            best_model = copy.deepcopy(model).cpu()

        print(f"epoch {epoch}. Train: {np.mean(errs_train)}")

    torch.save(best_model, helpers.get_exp_path() / f"{model_name}")

    ##open new tmux, type 16:8, wihtout activate .venv
    #Val MAE: 602.00390625
    #Val Corr: 0.8206432839493265
    #Val Loss: 0.13570427894592285
    #Steepness: 0.8459273945422541
