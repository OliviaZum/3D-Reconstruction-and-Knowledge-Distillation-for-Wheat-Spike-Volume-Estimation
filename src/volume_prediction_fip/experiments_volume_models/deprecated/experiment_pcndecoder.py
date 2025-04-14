import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import copy
import experiments.shared.utils_3d as utils_3d
import experiments.shared.models_3d as models_3d
import experiment_utils
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

def evaluate(dataloader: DataLoader, model: nn.Module):
    model = model.eval()
    emb, losses = fit_embeddings(dataloader, model, True, 50, True)
    print(f"Val loss: {losses[-1]}")
    if False:
        for idx, batch, _ in dataloader:
            batch = batch.to(device)
            idx = idx.to(device)
            x, _ = model(idx, emb)
            for i in range(len(batch)):
                utils_3d.visualize_point_clouds(batch[i, :, :3].detach().cpu().numpy(), x[i, :, :3].detach().cpu().numpy())
    return losses[-1]

if __name__ == "__main__":
    train_dataset, val_dataset = experiment_utils.get_ply_dataset()

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = models_3d.PointCloudAutoDecoder(128, len(train_loader.dataset)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, 0.5)

    best_val = None
    best_model = None
    for epoch in range(201):
        errs_train = []
        vol_losses_train = []
        model.train()
        for idx, batch, _ in train_loader:
            batch = batch.to(device)
            idx = idx.to(device)
            predictions, coarse_pred = model(idx)
            loss = utils_3d.chamfer_dist_slow_normal(predictions, batch, normal_loss=False)
            loss = loss + utils_3d.chamfer_dist_slow_normal(coarse_pred, batch, normal_loss=False)
            loss = loss + (model.embeddings.weight ** 2).mean()

            errs_train.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        if epoch % 30 == 0 and epoch > 0:
            err_val = evaluate(val_loader, model)

            if best_val is None or err_val < best_val:
                best_val = err_val
                best_model = copy.deepcopy(model).cpu()
        print(f"epoch {epoch}. Train: {np.mean(errs_train)}")
