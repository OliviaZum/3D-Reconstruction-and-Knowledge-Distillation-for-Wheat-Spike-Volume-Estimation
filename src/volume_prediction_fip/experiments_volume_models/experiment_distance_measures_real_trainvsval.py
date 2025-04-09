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
    train_dataset, val_dataset = experiment_utils.get_real_dataset()
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = torch.load("3dvol_approx/local_stuff/real_volume_model.pth", weights_only=False).to(device).eval()
    
    val_err, val_lat, val_pred, val_real = comp_features(val_loader, model)
    train_err, train_lat, train_pred, train_real = comp_features(train_loader, model)

    train_lat_np = train_lat.cpu().numpy()
    val_lat_np = val_lat.cpu().numpy()
    train_err_np = train_err.cpu().numpy()
    val_err_np = val_err.cpu().numpy()

    if False:
        mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42)
        feat2d = mds.fit_transform(np.concatenate((train_lat_np, val_lat_np)))
        train_2d = feat2d[:len(train_lat_np)]
        val_2d = feat2d[len(train_lat_np):]
        

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Scatter plot for training data
        sc_train = ax.scatter(
            train_2d[:, 0], train_2d[:, 1], c=train_err_np, cmap='coolwarm',
            marker='o', edgecolor='k', alpha=0.7, label="Train"
        )

        # Scatter plot for validation data
        sc_val = ax.scatter(
            val_2d[:, 0], val_2d[:, 1], c=val_err_np, cmap='coolwarm',
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

    if True:
        k_neighbors = 3

        lastlayer = model.lastlin._modules["0"].weight.cpu().detach().numpy()[0]
        def custom(v1, v2):
            h = np.abs(v1 - v2) * np.abs(lastlayer)    
            return h.sum()

        # Find nearest neighbors from train_lat for each val_lat
        nearest = NearestNeighbors(n_neighbors=k_neighbors, metric=custom).fit(train_lat_np)
        distances, indices = nearest.kneighbors(val_lat_np)

        # Compute mean distance to k nearest neighbors
        mean_distances = np.median(distances, axis=1)

        # Compute mean error of nearest neighbors
        mean_train_errors = train_err_np[indices].mean(axis=1)

        # Scatter plot
        fig, ax = plt.subplots(figsize=(8, 6))

        sc = ax.scatter(
            mean_distances, np.abs(val_err_np), c=np.abs(mean_train_errors), cmap='coolwarm',
            edgecolor='k', alpha=0.7
        )

        # Add colorbar
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Mean Train Neighbor Error")

        # Labels
        ax.set_title("Error vs Distance to Nearest Neighbors")
        ax.set_xlabel("Mean Distance to k Nearest Neighbors")
        ax.set_ylabel("Validation Error")

        # Show plot
        plt.show()