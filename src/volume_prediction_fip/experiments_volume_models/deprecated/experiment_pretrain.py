import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import copy
import experiments.shared.datasets_3d as datasets_3d
import experiments.shared.utils_3d as utils_3d
import experiment_utils
from torch.nn import functional as F
import matplotlib.pyplot as plt
import experiments.shared.models_3d as models_3d

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def evaluate(dataloader: DataLoader, model: nn.Module):
    model = model.eval()
    with torch.no_grad():
        losses = []
        for _, batch, _  in dataloader:
            batch = batch.to(device)
            batch = utils_3d.to_rigid_invariant_representation(batch[:, :, 0:3], batch[:, :, 6])

            complete_pred, _ = model(batch)
            loss = utils_3d.compare_rigid_invariant_mse(complete_pred, batch)
            losses.append(loss.cpu())
            
        loss = np.mean(losses)
        p97e = np.percentile(losses, 97)
        print(f"Val Loss (hist diff): {loss}")
        print(f"Val loss (variance): {np.var(losses)}")
        print(f"Val loss (p95): {p97e}")
        return p97e

if __name__ == "__main__":
    train_dataset, val_dataset = experiment_utils.get_unlabeled_dataset(False)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    encoder = models_3d.RigidInvariantPointNet(latent_size=128, output="latent").to(device)
    model = models_3d.RigidInvariantCompletion(encoder, 128).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, 0.5)

    best_val = None
    best_encoder = None
    for epoch in range(120):
        errs_train = []
        model.train()
        j = 0
        for idx, batch, _ in train_loader:
            batch = batch.to(device)
            batch = utils_3d.to_rigid_invariant_representation(batch[:, :, 0:3], batch[:, :, 6])
            complete_pred, latent = model(batch)

            loss = utils_3d.compare_rigid_invariant_mse(complete_pred, batch, average=True)

            errs_train.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        if True or epoch % 10 == 0 and epoch > 0:
            err_val = evaluate(val_loader, model)

            if best_val is None or err_val < best_val:
                best_val = err_val
                best_encoder = copy.deepcopy(encoder).cpu()

        print(f"epoch {epoch}. Train: {np.mean(errs_train)}")

    torch.save(best_encoder, "3dvol_approx/local_stuff/pretrained_model.pth")