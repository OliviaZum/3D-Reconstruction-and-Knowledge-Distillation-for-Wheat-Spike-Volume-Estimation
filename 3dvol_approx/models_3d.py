from torch import nn
import torch

# https://arxiv.org/pdf/1904.00069
class PCNEncoder(nn.Module):
    def __init__(self, latent_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        encfun = lambda x_in, x_out: nn.Sequential(
            nn.Conv1d(x_in, 64, 1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, 1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, x_out, 1),
            nn.AdaptiveMaxPool1d(1),
        )
        self.l1 = encfun(6, latent_size)
        self.l2 = encfun(6 + latent_size, latent_size)
    
    def forward(self, x: torch.Tensor):
        x = x.permute((0, 2, 1))
        x_in = x
        x = self.l1(x)
        x = x.repeat(1, 1, x_in.shape[2])
        x = torch.concat((x_in, x), dim=1)
        x = self.l2(x).squeeze()

        return x

# https://arxiv.org/pdf/1808.00671
class PCNDecoder(nn.Module):
    def __init__(self, latent_size: int = 128, point_cloud_size: int = 2000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.u = 5
        self.point_cloud_size = point_cloud_size
        assert point_cloud_size % self.u == 0
        self.coarse_size = point_cloud_size // (self.u ** 2)
        self.coarse = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.coarse_size * 6)
        )
        gx, gy = torch.meshgrid((torch.arange(0, self.u), torch.arange(0, self.u)), indexing="ij")
        g = torch.concat((gx.unsqueeze(2), gy.unsqueeze(2)), dim=2).unsqueeze(0)
        g = g.repeat(self.coarse_size, 1, 1, 1)
        self.grid = torch.flatten(g, start_dim=0, end_dim=2).unsqueeze(0)
        self.fine = nn.Sequential(
            nn.Linear(latent_size + 6 + 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 6)
        )

    def forward(self, x):
        coarse_pred = self.coarse(x)
        coarse_pred = coarse_pred.reshape((-1, self.coarse_size, 6))

        t = torch.repeat_interleave(coarse_pred, self.u ** 2, dim=1)
        xr = torch.repeat_interleave(x.unsqueeze(1), self.point_cloud_size, dim=1)
        g = torch.repeat_interleave(self.grid, x.shape[0], dim=0).to(x.device)
        c = torch.concat((t, g, xr), dim=2)
        fine_pred = self.fine(c) + t
        
        return fine_pred, coarse_pred
    
class PointCloudAutoEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder = PCNEncoder(128)
        self.decoder = PCNDecoder(128)
    
    def forward(self, x):
        latent = self.encoder(x)
        x = self.decoder(latent)
        if self.training:
            return x, latent
        else:
            return x
    
class PointCloudAutoEncoder_volume(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder = PCNEncoder(128)
        self.decoder = PCNDecoder(128)
        self.vol_pred = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        x = self.decoder(latent)
        volume = self.vol_pred(latent)
        if self.training:
            return volume, x, latent
        else:
            return volume, x
    

class PointCloudAutoDecoder(nn.Module):
    def __init__(self, latent_size, size_train, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_size = latent_size

        self.embeddings = nn.Embedding(size_train, latent_size)
        self.decoder = PCNDecoder(latent_size)

    def forward(self, idx, embeddings = None):
        if embeddings is not None:
            latent = embeddings(idx)
            """
            pose = latent[:, self.latent_size:]
            latent = latent[:, :self.latent_size]
            quat = pose[:, :4]
            quat = quat / torch.norm(quat, dim=1, keepdim=True)
            qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
            R = torch.stack([
                1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw),
                2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw),
                2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)
            ], dim=-1).view(-1, 3, 3)
            t = pose[:, 4:]
            """
        else:
            latent = self.embeddings(idx)
        if len(latent.shape) == 1:
            latent = latent.unsqueeze(0)

        coarse, fine = self.decoder(latent)
        """
        if embeddings is not None:
            ct = torch.matmul(coarse[:, :, :3], R) + t.unsqueeze(1)
            cn = torch.matmul(coarse[:, :, 3:], R)
            coarse = torch.concat((ct, cn), dim=2)
            ft = torch.matmul(fine[:, :, :3], R) + t.unsqueeze(1)
            fn = torch.matmul(fine[:, :, 3:], R)
            fine = torch.concat((ft, fn), dim=2)
        """
        return coarse, fine
    
class AffineInvariantPointNet(nn.Module):
    def __init__(self, k=160, bins=10, latent_size=128):
        super().__init__()
        self.k = k
        self.bins = bins
        encfun = lambda x_in, x_out: nn.Sequential(
            nn.Conv1d(x_in, 64, 1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, 1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, x_out, 1),
            nn.AdaptiveMaxPool1d(1),
        )
        self.l1 = encfun(bins, latent_size)
        self.l2 = encfun(bins + latent_size, latent_size)
        self.lastlin = nn.Linear(128, 1)
    
    def forward(self, x: torch.Tensor):
        u = torch.zeros((x.shape[0], self.k, self.bins), device=x.device)
        with torch.no_grad():
            w = self.distance_sample(x)
            u = self.batched_torch_hist(w, 0, 60, self.bins)
            u = u.to(x.dtype)
        x = u.permute((0, 2, 1))
        x_in = x
        x = self.l1(x)
        x = x.repeat(1, 1, x_in.shape[2])
        x = torch.concat((x_in, x), dim=1)
        x = self.l2(x).squeeze()
        x = self.lastlin(x)
        if self.training:
            return x, (None, None), None
        else:
            return x, None

    def distance_sample(self, points: torch.Tensor) -> torch.Tensor:
        batch, n, _ = points.shape
        sampled_indices = torch.randint(0, n, (batch, self.k), device=points.device).unsqueeze(-1).expand(-1, -1, points.shape[-1])
        sampled_points = torch.gather(points, 1, sampled_indices)
        
        # Compute pairwise L2 distances
        distances = torch.cdist(sampled_points, points, p=2)  # (k, n)
        return distances
    
    def batched_torch_hist(self, x: torch.Tensor, start: float, end: float, bins: int):
        bin_width = (end - start) / bins
        end = end - start
        x = x - start
        x = (x / bin_width).floor().to(torch.int32)
        bins = torch.arange(bins, device=x.device)
        mask_eq = x.unsqueeze(-1) == bins
        counts = mask_eq.sum(dim=-2)
        return counts