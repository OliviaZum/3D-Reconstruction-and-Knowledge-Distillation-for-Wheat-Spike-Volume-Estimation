import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import copy
import experiments.shared.utils_3d as utils_3d
import experiments.shared.models_3d as models_3d
import experiment_utils

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def evaluate(dataloader: DataLoader, model: nn.Module):
    model = model.eval()
    with torch.no_grad():
        losses = []
        for _, batch, _ in dataloader:
            batch = batch.to(device)
            x, _ = model(batch)
            loss = utils_3d.chamfer_dist_slow_normal(x, batch, normal_loss=False)
            losses.append(loss.item())
        losses = np.mean(losses)
        print(f"Val loss {losses}")
        if False:
            for idx, batch, _ in dataloader:
                batch = batch.to(device)
                x, _ = model(batch)
                for i in range(len(batch)):
                    utils_3d.visualize_point_clouds(batch[i, :, :3].detach().cpu().numpy(), x[i, :, :3].detach().cpu().numpy())
        return losses

if __name__ == "__main__":
    train_dataset, val_dataset = experiment_utils.get_ply_dataset()
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = models_3d.PointCloudAutoEncoder().to(device)
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
            (predictions, coarse_pred), latent = model(batch)
            loss = utils_3d.chamfer_dist_slow_normal(predictions, batch, normal_loss=False)
            loss = loss + utils_3d.chamfer_dist_slow_normal(coarse_pred, batch, normal_loss=False)
            loss = loss + (latent ** 2).mean()
            errs_train.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        if epoch % 2 == 0 and epoch > 0:
            err_val = evaluate(val_loader, model)

            if best_val is None or err_val < best_val:
                best_val = err_val
                best_model = copy.deepcopy(model).cpu()
        print(f"epoch {epoch}. Train: {np.mean(errs_train)}")