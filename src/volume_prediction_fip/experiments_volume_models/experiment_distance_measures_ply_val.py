import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import experiments.shared.datasets_3d as datasets_3d
import experiments.shared.utils_3d as utils_3d
import experiment_utils
from torch.nn import functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def comp_features(dataloader: DataLoader, model, show_plot = False):
    model = model.eval()
    with torch.no_grad():
        vol_pred = []
        vol_real = []
        latents = []
        for _, batch, vol_batch in dataloader:
            batch = batch.to(device)
            r = utils_3d.to_rigid_invariant_representation(batch[:, :, 0:3], batch[:, :, 6], num_samples=None)
            volume, latent = model(r)
            volume = volume.cpu().squeeze()
            vol_pred.append(volume)
            vol_real.append(vol_batch)
            latents.append(latent.cpu())
        vol_pred = torch.concat(vol_pred)
        vol_real = torch.concat(vol_real)
        latents = torch.concat(latents)

        loss = F.mse_loss(vol_pred, vol_real)
        print(f"Val MAE: {F.l1_loss(datasets_3d.DepthMapDataset.vol_unorm(vol_pred), datasets_3d.DepthMapDataset.vol_unorm(vol_real))}")
        print(f"Val Corr: {np.corrcoef(vol_real, vol_pred)[0, 1]}")
        print(f"Val Loss: {loss}")
        A = np.vstack([vol_real, np.ones(len(vol_real))]).T
        linregcoeff, _, _, _ = np.linalg.lstsq(A, vol_pred, rcond=None)
        print(f"Steepness: {linregcoeff[0]}")
        return vol_pred - vol_real, latents, vol_pred, vol_real

if __name__ == "__main__":
    _, val_dataset = experiment_utils.get_real_dataset()
    train_ply, _ = experiment_utils.get_ply_dataset()
    train_ply_loader = DataLoader(train_ply, batch_size=16, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model_ply = torch.load("3dvol_approx/local_stuff/ply2volume_model.pth", weights_only=False).to(device).eval()
    model_real = torch.load("3dvol_approx/local_stuff/real_volume_model.pth", weights_only=False).to(device).eval()
        
    _, val_lat, _, _ = comp_features(val_loader, model_ply)
    _ , ply_lat, _, _ = comp_features(train_ply_loader, model_ply)
    val_err, _, _, _ = comp_features(val_loader, model_real)

    ply_lat_np = ply_lat.cpu().numpy()
    val_lat_np = val_lat.cpu().numpy()
    val_err_np = val_err.cpu().numpy()

    if True:
        mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42)
        feat2d = mds.fit_transform(np.concatenate((ply_lat_np, val_lat_np)))
        train_2d = feat2d[:len(ply_lat_np)]
        val_2d = feat2d[len(ply_lat_np):]

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Scatter plot for training data
        sc_train = ax.scatter(
            train_2d[:, 0], train_2d[:, 1], 
            marker='o', edgecolor='k', alpha=0.7, label="Ply objects"
        )

        # Scatter plot for validation data
        sc_val = ax.scatter(
            val_2d[:, 0], val_2d[:, 1], c=np.abs(val_err_np), cmap='coolwarm',
            marker='s', edgecolor='k', alpha=0.7, label="Validation"
        )

        # Add colorbar
        cbar = plt.colorbar(sc_train, ax=ax)
        cbar.set_label("Error")

        # Labels & Legend
        ax.set_title("MDS Visualization with Error Coloring")
        ax.set_xlabel("MDS Dim 1")
        ax.set_ylabel("MDS Dim 2")
        ax.legend()

        # Show plot
        plt.show()

    if False:
        k_neighbors = 50

        # Find nearest neighbors from train_lat for each val_lat
        nearest = NearestNeighbors(n_neighbors=k_neighbors, metric="manhattan").fit(ply_lat_np)
        distances, _ = nearest.kneighbors(val_lat_np)

        # Compute mean distance to k nearest neighbors
        mean_distances = np.median(distances, axis=1)

        # Scatter plot
        fig, ax = plt.subplots(figsize=(8, 6))

        sc = ax.scatter(
            mean_distances, np.abs(val_err_np),
            edgecolor='k', alpha=0.7
        )

        # Labels
        ax.set_title("Error vs Distance to Nearest Neighbors")
        ax.set_xlabel("Mean Distance to k Nearest Neighbors")
        ax.set_ylabel("Validation Error")

        # Show plot
        plt.show()