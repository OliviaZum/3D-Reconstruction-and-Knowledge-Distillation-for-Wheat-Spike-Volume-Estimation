# experiment_point_transformer_v3.py

import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import copy
from torch.nn import functional as F
import matplotlib.pyplot as plt
import accelerate
from sklearn.decomposition import PCA
from volume_prediction_fip.experiments_volume_models.shared import utils_experiment, datasets
from volume_prediction_fip.utils import helpers
from mpl_toolkits.mplot3d import Axes3D

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# --------------------------------------------------
# Real Geometry-Aware Point Transformer Layer
# --------------------------------------------------

class PointTransformerLayer(nn.Module):
    def __init__(self, dim, k=16):
        super().__init__()
        self.k = k

        # feature projections
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        # relative position encoding
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        # attention score MLP
        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        # feed-forward block
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, coords):
        # x: (B, N, C)
        # coords: (B, N, 3)

        B, N, C = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # pairwise distances
        dist = torch.cdist(coords, coords)  # (B, N, N)

        # kNN indices
        knn_idx = dist.topk(self.k, largest=False).indices  # (B, N, k)

        # expand tensors for gathering
        k_expand = k.unsqueeze(1).expand(-1, N, -1, -1)
        v_expand = v.unsqueeze(1).expand(-1, N, -1, -1)
        coord_expand = coords.unsqueeze(1).expand(-1, N, -1, -1)

        # gather neighbors
        k_neighbors = torch.gather(
            k_expand,
            2,
            knn_idx.unsqueeze(-1).expand(-1, -1, -1, C)
        )

        v_neighbors = torch.gather(
            v_expand,
            2,
            knn_idx.unsqueeze(-1).expand(-1, -1, -1, C)
        )

        coord_neighbors = torch.gather(
            coord_expand,
            2,
            knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        )

        # relative positional encoding
        relative_pos = coords.unsqueeze(2) - coord_neighbors  # (B, N, k, 3)
        delta = self.pos_mlp(relative_pos)  # (B, N, k, C)

        # attention scores (scalar per neighbor)
        attn = self.attn_mlp(q.unsqueeze(2) - k_neighbors + delta)
        attn = attn.mean(dim=-1, keepdim=True)  # scalar weight
        attn = torch.softmax(attn, dim=2)

        # weighted aggregation
        out = torch.sum(attn * (v_neighbors + delta), dim=2)

        # residual + normalization
        x = self.norm1(x + out)

        # feed-forward block
        x = self.norm2(x + self.ffn(x))

        return x


# --------------------------------------------------
# Point Transformer Regression Model
# --------------------------------------------------

class PointTransformerRegressor(nn.Module):
    def __init__(self, input_dim=6, embed_dim=128, depth=3, k=16):
        super().__init__()

        self.embedding = nn.Linear(input_dim, embed_dim)

        self.layers = nn.ModuleList([
            PointTransformerLayer(embed_dim, k=k)
            for _ in range(depth)
        ])

        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, x):
        coords = x[:, :, :3]   # keep geometry
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x, coords)

        x = x.mean(dim=1)  # global pooling
        x = self.head(x)

        return x, None



# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

def preproc_batch(batch):
    #with torch.no_grad():
    #    # Keep exact same normalization as PointNet
    #    batch[:, :, 0:3] = batch[:, :, 0:3] / 60
    #    return batch[:, :, 0:6]  # [B, N, 6]
    
    with torch.no_grad():
        # Center per spike
        coords = batch[:, :, 0:3]
        coords = coords - coords.mean(dim=1, keepdim=True)

        # Scale per spike
        scale = coords.norm(dim=2).max(dim=1)[0].view(-1, 1, 1)
        coords = coords / (scale + 1e-8)

        batch[:, :, 0:3] = coords
        return batch[:, :, 0:6]  # [B, N, 6]


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate(dataloader, model, show_plot=False):
    model.eval()
    with torch.no_grad():
        vol_pred = []
        vol_real = []

        for _, batch, vol_batch, mask, weight in dataloader:
            batch = batch.to(device)
            vol_batch = vol_batch.to(device)

            r = preproc_batch(batch)
            volume, _ = model(r)

            vol_pred.append(volume.squeeze().cpu())
            vol_real.append(vol_batch.cpu())

        vol_pred = torch.cat(vol_pred)
        vol_real = torch.cat(vol_real)

        print(f"Val MAE: {F.l1_loss(datasets.vol_unorm(vol_pred), datasets.vol_unorm(vol_real))}")

        eps = 1e-8
        vol_pred_un = datasets.vol_unorm(vol_pred)
        vol_real_un = datasets.vol_unorm(vol_real)
        mape = torch.mean(torch.abs((vol_real_un - vol_pred_un) / (vol_real_un + eps))) * 100
        print(f"Val MAPE: {mape}")
        
        print(f"Val Corr: {np.corrcoef(vol_real, vol_pred)[0, 1]}")

        A = np.vstack([vol_real, np.ones(len(vol_real))]).T
        linregcoeff, _, _, _ = np.linalg.lstsq(A, vol_pred, rcond=None)
        print(f"Steepness: {linregcoeff[0]}")

        if show_plot:
            plt.scatter(datasets.vol_unorm(vol_real), datasets.vol_unorm(vol_pred))
            plt.plot((2000, 8500), (2000, 8500))
            plt.show()

        return F.mse_loss(vol_pred, vol_real).item()
    

# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    accelerate.utils.set_seed(1, deterministic=True)

    train_dataset, val_dataset, test_dataset = utils_experiment.get_real_dataset3d(force_recompute=False)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    evaluation = True
    if evaluation: 
        model = torch.load(
            helpers.get_exp_path() / "point_transformer_v3.pth",
            weights_only=False
        ).to(device).eval()

        plot_pca = True
        if plot_pca:
            print("Final Test PCA:")
            best_model = model.to(device).eval()

            all_features = []
            all_volumes = []

            with torch.no_grad():
                for _, batch, vol_batch, mask, weight in test_loader:

                    batch = batch.to(device)
                    r = preproc_batch(batch)

                    # ---- Forward until global pooling ----
                    coords = r[:, :, :3]
                    x = best_model.embedding(r)

                    for layer in best_model.layers:
                        x = layer(x, coords)

                    features = x.mean(dim=1)

                    all_features.append(features.cpu())
                    all_volumes.append(vol_batch.cpu())

            features = torch.cat(all_features)
            volumes = torch.cat(all_volumes)

            print("Feature shape:", features.shape)

            pca = "2D"

            if pca == "2D":
                pca = PCA(n_components=2)
                lat2d = pca.fit_transform(features.numpy())

                vol_mm3 = datasets.vol_unorm(volumes).numpy()

                fig, ax = plt.subplots(figsize=(6,5))

                sc = ax.scatter(
                    lat2d[:, 0],
                    lat2d[:, 1],
                    c=vol_mm3,
                    cmap="viridis",
                    s=30
                )

                cbar = fig.colorbar(sc, ax=ax)
                cbar.set_label("True volume [mm³]", fontsize=16)
                #cbar.ax.tick_params(labelsize=15)

                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")

                ax.tick_params(axis='both')

                ax.set_xlim(-12, 12)
                ax.set_ylim(-12, 12)

                fig.tight_layout()

                fig.savefig(
                    helpers.get_exp_path() / "pca_point_transformer_v3.png",
                    dpi=300
                )

                corr_pc1 = np.corrcoef(lat2d[:,0], volumes.numpy())[0,1]
                corr_pc2 = np.corrcoef(lat2d[:,1], volumes.numpy())[0,1]

                print("Correlation PC1 vs normalized volume:", corr_pc1)
                print("Correlation PC2 vs normalized volume:", corr_pc2)

            elif pca == "3D":
                
                pca = PCA(n_components=3)
                lat3d = pca.fit_transform(features.numpy())

                vol_mm3 = datasets.vol_unorm(volumes).numpy()
                
                fig = plt.figure(figsize=(7,6))
                ax = fig.add_subplot(111, projection='3d')

                sc = ax.scatter(
                    lat3d[:,0],
                    lat3d[:,1],
                    lat3d[:,2],
                    c=vol_mm3,
                    cmap="viridis",
                    s=30
                )


                fig.colorbar(sc, label="True volume [mm³]")

                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.set_zlabel("PC3")
                ax.view_init(elev=20, azim=45)

                plt.savefig(helpers.get_exp_path() / "pca3d_point_transformer_v3.png", dpi=600)

        else:
            print("Final Test:")
            evaluate(test_loader, model.to(device), show_plot=True)

        exit(0)

    model = PointTransformerRegressor().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.5)

    nepoch = 60

    best_val = None
    best_model = None

    for epoch in range(nepoch):
        model.train()
        errs = []

        for idx, batch, vol_batch, mask, weight in train_loader:
            batch = batch.to(device)
            vol_batch = vol_batch.to(device)

            r = preproc_batch(batch)
            volume, _ = model(r)

            loss = F.mse_loss(volume.squeeze(), vol_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            errs.append(loss.item())

        scheduler.step()

        val_loss = evaluate(val_loader, model)

        if best_val is None or val_loss < best_val:
            best_val = val_loss
            best_model = copy.deepcopy(model).cpu()

        print(f"Epoch {epoch} | Train: {np.mean(errs)}")

    torch.save(best_model, helpers.get_exp_path() / "point_transformer_v3.pth")
    


#with centering and scaling: 
#Final Test:
#Val MAE: 728.1543579101562
#Val Corr: 0.6373971014946634
#Steepness: 0.36821130400527224

#without centering and scaling:
#Val MAE: 742.9132690429688
#Val MAPE: 16.708850860595703
#Val Corr: 0.6236517307551169
#Steepness: 0.3444992723954968