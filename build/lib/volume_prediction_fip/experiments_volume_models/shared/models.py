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
    def __init__(self, latent_size=128, bins=20, output: Literal["volume", "latent", "volumevar", "volumelatent"] = "volume"):
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
        )
        self.l1 = encfun(bins, latent_size)
        self.output = output
        if self.output != "latent":
            flast = lambda: nn.Sequential(
                nn.Linear(latent_size, latent_size),
                nn.Dropout(0.3),
                nn.GELU(),
                nn.Linear(latent_size, 1)
            )
            self.lastlin = flast()
            self.lastvar = flast()
            

    def compute_mean(self, latent, mask):
        msum = mask.sum(dim=-1).reshape((-1, 1, 1))
        msum[msum == 0] = 1

        latent = latent.permute((0, 2, 1))
        latent[~mask, :] = 0
        latent = torch.sum(latent, dim=1, keepdim=True) / msum
        latent = latent.permute((0, 2, 1))

        return latent
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor, output=None):


        x = x.permute((0, 2, 1))
        features = self.l1(x)
        features = self.compute_mean(features, mask).squeeze()

        if output == "features":
            return features

        if self.output == "volume" or self.output == "volumelatent":
            v = self.lastlin(features)
            if self.output == "volume":
                return v
            else:
                return v, features
        elif self.output == "volumevar":
            vol = self.lastlin(features)
            var = self.lastvar(features)
            return vol, var
        else:
            return features
    
class RigidInvariantPointNet2Layered(nn.Module):
    # Overfits
    def __init__(self, latent_size=128, bins=10, output: Literal["volume", "latent"] = "volume"):
        super().__init__()
        encfunup = lambda x_in, x_out: nn.Sequential(
            nn.Conv1d(x_in, x_out, 1),
        )
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
        self.l1up = encfunup(bins, latent_size)
        self.l2 = encfun(bins + latent_size, latent_size)
        self.l2up = encfunup(bins + latent_size, latent_size)
        self.output = output
        if self.output == "volume":
            self.lastlin = nn.Sequential(
                nn.Linear(latent_size, latent_size),
                nn.Dropout(0.3),
                nn.GELU(),
                nn.Linear(latent_size, 1)
            )

    def compute_mean(self, latent, mask):
        msum = mask.sum(dim=-1).reshape((-1, 1, 1))
        msum[msum == 0] = 1

        latent = latent.permute((0, 2, 1))
        latent[~mask, :] = 0
        latent = torch.sum(latent, dim=1, keepdim=True) / msum
        latent = latent.permute((0, 2, 1))

        return latent
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x = x.permute((0, 2, 1))
        x_in = x
        latent = self.l1(x)
        short = self.l1up(x)
        latent = self.compute_mean(latent + short, mask)

        x = latent.expand(-1, -1, x_in.shape[2])
        x = torch.concat((x_in, x), dim=1)
        latent = self.l2(x)
        short = self.l2up(x)
        latent = self.compute_mean(latent + short, mask).squeeze()

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
    
    def forward(self, x, mask):
        latent = self.encoder(x, mask)
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

def unflat(shape, mask, input):
    unflat_array = torch.zeros(shape, device=input.device, dtype=input.dtype)
    unflat_array[~mask] = input
    return unflat_array

class RegulatedTransformer(nn.Module):
    def __init__(self, output: Literal["features", "volume"] = "volume", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.subdim = 192

        fmlp1 = lambda: nn.Sequential(
            nn.Linear(384, 384),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(384, self.subdim),
        )
        fmlp2 = lambda: nn.Sequential(
            nn.Linear(self.subdim, 1)
        )
        fmlp3 = lambda: nn.Sequential(
            nn.Linear(self.subdim * 2, self.subdim * 2),
            nn.GELU(),
            nn.Linear(self.subdim * 2, 1)
        )
        self.subdim_pred = fmlp1()
        self.subdim_conf = fmlp1()
        self.head_pred = fmlp2()
        self.head_conf = fmlp2()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.subdim * 2,
            nhead=2,
            dim_feedforward=4 * self.subdim,
            dropout=0.5,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=4, norm=None
        )
        self.volume_token = nn.Parameter(torch.randn(1, 1, self.subdim * 2))
        self.output = output
        if self.output == "volume":
            self.last_pred = fmlp3()
            self.last_conf = fmlp3()

        self.kd_proj = nn.Linear(self.subdim * 2, self.subdim * 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, return_features=False, output=None):

        if output == "features":
            return_features = True

        
        pred_sdim: torch.Tensor = self.subdim_pred(x[~mask]) #
        conf_sdim = self.subdim_conf(x[~mask])

        direct_pred = self.head_pred(pred_sdim) #mean prediction
        direct_conf = self.head_conf(conf_sdim) #log variance prediction, (n_valid, 1), squeeze makes (n_valid)

        #mask: 2D? (batch, n_images), true if empty
        direct_pred_unflat = unflat((x.shape[0], x.shape[1]), mask, direct_pred.squeeze())
        direct_conf_unflat = unflat((x.shape[0], x.shape[1]), mask, direct_conf.squeeze()) #target shape: (batch, n_images), 2D tensor
        pred_sdim_unflat = unflat((x.shape[0], x.shape[1], self.subdim), mask, pred_sdim)
        conf_sdim_unflat = unflat((x.shape[0], x.shape[1], self.subdim), mask, conf_sdim)# (n_valid, 192) --> (batch, n_images, 192)
        conf_sdim_unflat = unflat((x.shape[0], x.shape[1], self.subdim), mask, conf_sdim)

        pred_conf_cat = torch.cat((pred_sdim_unflat, conf_sdim_unflat), dim=2)
        transformer_input = torch.cat((pred_conf_cat, self.volume_token.repeat(x.shape[0], 1, 1)), dim=1) #add volume token (batch, 1, 284)
        mask_n = torch.zeros((mask.shape[0], mask.shape[1] + 1), dtype=torch.bool, device=mask.device)
        mask_n[:, -1] = 0
        mask_n[:, 0:-1] = mask

        transformer_output = self.encoder(src=transformer_input, src_key_padding_mask=mask_n)
        vol_pred_token = transformer_output[:, -1]

        if return_features:
            return vol_pred_token

        if self.output == "volume":
            tpred = self.last_pred(vol_pred_token)
            tconf = self.last_conf(vol_pred_token)

        if self.training and self.output == "volume":
            return direct_pred_unflat, direct_conf_unflat, tpred, tconf
        elif self.output == "features":
            return vol_pred_token
        else:
            return tpred, tconf
    
class RegulatedTransformerLoss(nn.Module):
    def __init__(self, weights = (1, 0.5), *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.weights = weights

    def forward(self, input: torch.Tensor, target: torch.Tensor, loss_scale: torch.Tensor) -> torch.Tensor:
        imgmeans, logimgvars, mean, logvars = input
        mask = imgmeans != 0

        target = target.unsqueeze(1)
        loss_scale = loss_scale.unsqueeze(1)

        err_single_img = ((imgmeans - target) ** 2) / (2 * torch.exp(logimgvars)) + 0.5 * logimgvars
        err_single_img = (err_single_img * loss_scale)[mask]
        
        err_combined = ((mean - target) ** 2 / (2 * torch.exp(logvars)) + 0.5 * logvars) * loss_scale

        wsi, wec = self.weights
        return err_single_img.mean() * wsi + err_combined.mean() * wec

class Image3dEnsemble(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_net = RegulatedTransformer(output="features")
        self.point_net = RigidInvariantPointNet(output="latent", bins=10)

        self.prec_img = nn.Sequential(
            nn.Linear(384, 384),
            nn.GELU(),
            nn.Linear(384, 128)
        )


        self.final = nn.Sequential(
            nn.Linear(256, 256),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(256, 1)
        )

    def forward(self, images, images_mask, points, point_mask, return_features=False):
        imgfeat = self.img_net(images, images_mask, output="features")
        imgfeat = self.prec_img(imgfeat)
        pointfeat = self.point_net(points, point_mask, output="features")

        combined = torch.concat((imgfeat, pointfeat), dim=-1)

        if return_features:
            return combined
        
        pred = self.final(combined)
        return pred
    
class SingleImageModel(nn.Module):
    def __init__(self, output: Literal["normal", "features"] = "normal", *args, **kwargs):
        super().__init__(*args, **kwargs)
        fmlp = lambda: nn.Sequential(
            nn.Linear(384, 384),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(384, 192),
            nn.LeakyReLU(),
        )
        self.last_mean = nn.Linear(192, 1)
        self.last_var = nn.Linear(192, 1)
        self.mean_l = fmlp()
        self.var_l = fmlp()
        self.output = output

    def forward(self, x, mask):
        unflat_shape = (x.shape[0], x.shape[1])
        unflat_shape_features = (*unflat_shape, 192)
        x = x[~mask]
        meanfeatures = self.mean_l(x)
        varfeatures = self.var_l(x)
        
        if self.output == "normal":
            mean = self.last_mean(meanfeatures)
            var = self.last_var(varfeatures)
            mean = unflat(unflat_shape, mask, mean.squeeze())
            var = unflat(unflat_shape, mask, var.squeeze())
            return mean, var
        else:
            meanfeatures = unflat(unflat_shape_features, mask, meanfeatures)
            varfeatures = unflat(unflat_shape_features, mask, varfeatures)
            return meanfeatures, varfeatures
        
class AttentionBased(nn.Module):
    
    def __init__(self, 
                 dropout_rate : float = 0.5,
                 layer: int = 4,
                 head: int = 2,
                 dim_forward: int = 4 * 384):
        super(AttentionBased, self).__init__()
        
        self.single_img_mlp = nn.Linear(384, 384)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=384,
            nhead=head,
            dim_feedforward=dim_forward,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=layer, norm=None
        )

        self.volume_token = nn.Parameter(torch.randn(1, 1, 384))

        self.last_lin = nn.Sequential(
            nn.Linear(384, 384),
            nn.GELU(),
            nn.Linear(384, 1)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        
        x = self.single_img_mlp(x)

        x = torch.cat((x, self.volume_token.repeat(x.size(0), 1, 1)), dim=1)
        mask_n = torch.zeros((mask.shape[0], mask.shape[1] + 1), dtype=torch.bool, device=mask.device)
        mask_n[:, -1] = 0
        mask_n[:, 0:-1] = mask

        x = self.encoder(src=x, src_key_padding_mask=mask_n)
        x = x[:, -1]

        x = self.last_lin(x)
        
        return x
    
class LSTM_fixed_dimensions(nn.Module): 
    def __init__(self,
                activation : nn.Module = nn.CELU,
                output_size : int = 1, 
                normalize : nn.Module = nn.LayerNorm,
                dropout_rate : float = 0.5): 
        super(LSTM_fixed_dimensions, self).__init__()
        input_size = 384
        hidden_size = 512
        num_layers = 3
        bidirectional = False
        num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout_rate)
        
        self.norm = normalize(hidden_size * num_directions)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = activation()
        self.output_layer = nn.Linear(hidden_size * num_directions, output_size)

        
    def forward(self, x : torch.Tensor, mask : torch.Tensor = None):
        assert x.dim() == 3, "Wrong input dimension, consider reshaping!" # batch x sequence x features
        
        if not mask is None:
            x = torch.einsum('jkl,jk->jkl',x, ~mask)

        lstm_out, _ = self.lstm(x)
        lstm_out_last = lstm_out[:, -1, :]
        x = self.dropout(self.activation(self.norm(lstm_out_last)))
        x = self.output_layer(x)

        return x