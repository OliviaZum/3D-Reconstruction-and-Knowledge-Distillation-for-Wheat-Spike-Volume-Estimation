from typing import Literal
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
    
class RigidInvariantPointNet(nn.Module):
    def __init__(self, latent_size=128, bins=10, output: Literal["volume", "latent"] = "volume"):
        super().__init__()
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
            #nn.AdaptiveAvgPool1d(1),
        )
        self.l1 = encfun(bins, latent_size)
        self.l2 = encfun(bins + latent_size, latent_size)
        self.output = output
        if self.output == "volume":
            self.lastlin = nn.Sequential(
                nn.Linear(latent_size, 1)
            )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x = x.permute((0, 2, 1))
        x_in = x
        latent = self.l1(x).squeeze()
        msum = mask.sum(dim=-1).unsqueeze(1)
        msum[msum == 0] = 1
        latent = latent.permute((0, 2, 1))
        latent[~mask, :] = 0
        latent = torch.sum(latent, dim=1) / msum
        """
        x = x.expand(-1, -1, x_in.shape[2])
        x = torch.concat((x_in, x), dim=1)
        x = self.l2(x).squeeze()
        """
        if self.output == "volume":
            x = self.lastlin(latent)
        return x, latent
    
class RigidInvariantCompletion(nn.Module):
    def __init__(self, encoder, latent_size = 128, point_cloud_size_output = 1000, bins = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 2 * latent_size),
            nn.ReLU(),
            nn.Linear(2 * latent_size, point_cloud_size_output * bins),
        )
        self.point_cloud_size_output = point_cloud_size_output
        self.bins = bins
    
    def forward(self, x):
        latent = self.encoder(x)
        x = self.decoder(latent)
        x = x.reshape(-1, self.point_cloud_size_output, self.bins)
        return x, latent
    
class RigidInvariantTr(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dim = 400
        self.preproc = nn.Sequential(
            nn.Linear(10, dim)
        )
        layer = nn.TransformerEncoderLayer(dim, 2, 800, batch_first=True)
        self.tr = nn.TransformerEncoder(layer, 2, norm=None)
        self.last = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x):
        x = self.preproc(x)
        x = torch.concat((x, self.cls_token.expand(x.shape[0], -1, -1)), dim=1)
        x = self.tr(x)
        cls = x[:, -1, :]
        x = x[:, :-1, :].mean(dim=1) + cls

        x = self.last(x)
        return x, None
