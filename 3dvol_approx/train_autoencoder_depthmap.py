import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import copy
import datasets_3d
import utils_3d
import models_3d
from pathlib import Path
from torch.nn import functional as F

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def fit_embeddings(loader, model3d, verbose=True, epochs=300, get_losses = False):
    embeddings = nn.Embedding(len(loader.dataset), 128).to(device)
    model3d = model3d.to(device).eval()
    optimizer = torch.optim.Adam(embeddings.parameters(), 0.01)
    losses_glob = []
    for e in range(epochs):
        losses = []
        for idx, batch, _ in loader:
            batch = batch.to(device)
            idx = idx.to(device)
            x, _ = model3d(idx, embeddings)
            loss = utils_3d.chamfer_dist_slow_normal(x, batch, normal_loss=False)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses_glob.append(np.mean(losses))
        if verbose:
            print(f"Epoch {e}: {losses_glob[-1]}; ", end="")
    if verbose:
        print("\n")
    if get_losses:
        return embeddings, losses_glob
    else:
        return embeddings

def evaluate(dataloader: DataLoader, model: nn.Module, decoder_only: bool, volume_direct: bool):
    model = model.eval()
    with torch.no_grad():
        if decoder_only:
            emb, losses = fit_embeddings(dataloader, model, True, 50, True)
        elif not volume_direct:
            losses = []
            for _, batch, _ in dataloader:
                batch = batch.to(device)
                x, _ = model(batch)
                loss = utils_3d.chamfer_dist_slow_normal(x, batch, normal_loss=False)
                losses.append(loss.item())
            losses = [np.mean(losses)]
        else:
            vol_pred = []
            vol_real = []
            for _, batch, vol_batch in dataloader:
                batch = batch.to(device)
                volume, _ = model(batch)
                volume = volume.cpu().squeeze()
                vol_pred.append(volume)
                vol_real.append(vol_batch)
            vol_pred = torch.concat(vol_pred)
            vol_real = torch.concat(vol_real)
            losses = [F.mse_loss(vol_pred, vol_real)]
            print(f"MAE: {F.l1_loss(datasets_3d.DepthMapDataset.vol_unorm(vol_pred), datasets_3d.DepthMapDataset.vol_unorm(vol_real))}")
        if False:
            for idx, batch, _ in dataloader:
                batch = batch.to(device)
                if decoder_only:
                    idx = idx.to(device)
                    x, _ = model(idx, emb)
                else:
                    x, _ = model(batch)
                for i in range(len(batch)):
                    utils_3d.visualize_point_clouds(batch[i, :, :3].detach().cpu().numpy(), x[i, :, :3].detach().cpu().numpy())
        return losses[-1]

def train(split_folder: Path, out: Path, img_base_folder : Path = None, decoder_only = False, real_data = True, volume_direct = True):
    #train_dataset = PlyDataset("3dvol_approx/train_ply.csv", "3dvol_approx/train_cache.pth", True)
    #val_dataset = PlyDataset("3dvol_approx/val_ply.csv", "3dvol_approx/val_cache.pth", True)
    if volume_direct:
        assert real_data
    if real_data:
        train_dataset = datasets_3d.DepthMapDataset(img_base_folder, img_base_folder / split_folder / "mapping_train.json", min_seq_len=6)
        train_dataset.create_or_load_cache(Path(r"3dvol_approx\local_stuff\dmap_cache\dmap_cache_train.pth"), force_recompute=False)
        val_dataset = datasets_3d.DepthMapDataset(img_base_folder, img_base_folder / split_folder / "mapping_val.json", augment=False)
        val_dataset.create_or_load_cache(Path(r"3dvol_approx\local_stuff\dmap_cache\dmap_cache_val.pth"), force_recompute=False)
    else:
        train_dataset = datasets_3d.PlyDataset(split_folder / "train_ply.csv", Path(r"3dvol_approx\local_stuff\ply_cache\ply_train_cache.pth"), True, augment=True)
        val_dataset = datasets_3d.PlyDataset(split_folder / "val_ply.csv", Path(r"3dvol_approx\local_stuff\ply_cache\ply_val_cache.pth"), True)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    if decoder_only:
        model = models_3d.PointCloudAutoDecoder(128, len(train_loader.dataset)).to(device)
    elif not volume_direct:
        model = models_3d.PointCloudAutoEncoder().to(device)
        #model = torch.load(out).to(device)
        #evaluate(val_loader, model, False)
        #exit(0)
    else:
        #model = models_3d.PointCloudAutoEncoder_volume().to(device)
        model = models_3d.AffineInvariantPointNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, 0.5)

    best_val = None
    best_model = None
    for epoch in range(201):
        errs_train = []
        vol_losses_train = []
        model.train()
        for idx, batch, vol_batch in train_loader:
            batch = batch.to(device)
            if decoder_only:
                idx = idx.to(device)
                predictions, coarse_pred = model(idx)
            elif not volume_direct:
                (predictions, coarse_pred), latent = model(batch)
                #utils_3d.visualize_point_clouds(batch[0, :, :3].detach().cpu().numpy(), predictions[0, :, :3].detach().cpu().numpy())
            else:
                volume, (predictions, coarse_pred), latent = model(batch[:, :, 0:3])
            #loss = utils_3d.chamfer_dist_slow_normal(predictions, batch, normal_loss=False)
            #loss = loss + utils_3d.chamfer_dist_slow_normal(coarse_pred, batch, normal_loss=False)
            """
            loss = torch.tensor(0)
            if decoder_only:
                loss = loss + (model.embeddings.weight ** 2).mean()
            else:
                loss = loss + (latent ** 2).mean()
            """
            
            if volume_direct:
                vol_batch = vol_batch.to(device)
                loss_vol = F.mse_loss(volume.squeeze(), vol_batch)
                vol_losses_train.append(loss_vol.item())
                loss = loss_vol# + loss

            errs_train.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        if epoch % 2 == 0 and epoch > 0:
            err_val = evaluate(val_loader, model, decoder_only, volume_direct)

            if best_val is None or err_val < best_val:
                best_val = err_val
                best_model = copy.deepcopy(model).cpu()

            print(f"epoch {epoch}. Val: {err_val}, Train: {np.mean(errs_train)}")
        else:
            print(f"epoch {epoch}. Train: {np.mean(errs_train)}")
        if volume_direct:
            print(f"vol loss train: {np.mean(vol_losses_train)}")

    torch.save(best_model, out)

def export_latents(split_folder: Path, out: Path, img_base_folder : Path = None, decoder_only = False, real_data = True):
    pass

if __name__ == "__main__":
    train(Path("split_without2024"),
          Path(r"C:\Users\Admin\Desktop\master_thesis\volume_prediction_fip\3dvol_approx\local_stuff\real_encoder_noaffine.pth"),
          Path(r"F:\Boxes-ds\segmented_distance_depth"), real_data=True, volume_direct=True)
    #train(Path(r"3dvol_approx\local_stuff\ply_split"), 
    #      Path(r"C:\Users\Admin\Desktop\master_thesis\volume_prediction_fip\3dvol_approx\local_stuff\arti_encoder_noaffine.pth"), real_data=False)

    #train(Path("split_without2024"),
    #      Path(r"C:\Users\Admin\Desktop\master_thesis\volume_prediction_fip\3dvol_approx\local_stuff\arti_encoder_noaffine.pth"),
    #      Path(r"F:\Boxes-ds\segmented_distance_depth"), real_data=True)