import experiments.shared.utils_experiment as utils_experiment
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import copy
import experiments.shared.datasets_3d as datasets_3d
import experiments.shared.utils_3d as utils_3d
import experiments.shared.models_3d as models_3d
import experiments.shared.utils_experiment as utils_experiment
from torch.nn import functional as F
import matplotlib.pyplot as plt
import pandas as pd
import torch
import itertools
import accelerate

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def evaluate(dataloader: DataLoader, model: nn.Module, show_plot = False):
    with torch.no_grad():
        vol_pred = []
        vol_real = []
        vol_std = []
        weights = []
        for images, points, imagemask, pointmask, label, weight in dataloader:
            images, points, imagemask, pointmask = [t.to(device) for t in [images, points, imagemask, pointmask]]
            points = utils_3d.to_rigid_invariant_representation(points[:, :, 0:3], points[:, :, 6], num_samples=None)
            volume, error = model(images, imagemask, points, pointmask)
            volume = volume.cpu().squeeze()

            vol_real.append(label)
            weights.append(weight)
        vol_pred = torch.concat(vol_pred)
        vol_real = torch.concat(vol_real)
        weights = torch.concat(weights)
        vol_std = torch.concat(vol_std)

        mae = F.l1_loss(datasets_3d.vol_unorm(vol_pred), datasets_3d.vol_unorm(vol_real))
        print(f"Val MAE: {mae}")
        corr = np.corrcoef(vol_real, vol_pred)[0, 1]
        print(f"Val Corr: {corr}")
        print(f"Val Loss: {((vol_pred - vol_real) ** 2 * weights).mean().item()}")
        A = np.vstack([vol_real, np.ones(len(vol_real))]).T
        linregcoeff, _, _, _ = np.linalg.lstsq(A, vol_pred, rcond=None)
        print(f"Steepness: {linregcoeff[0]}")
        rate = -(- mae + corr * (50 / 0.03) - abs(1 - linregcoeff[0]) * (50 / 0.1))
        print(f"Rate: {rate}")
        if show_plot:
            plt.plot((2000, 8500), (2000, 8500))
            plt.scatter(datasets_3d.vol_unorm(vol_real), datasets_3d.vol_unorm(vol_pred))
            plt.show()
            plt.scatter(torch.abs(vol_real - vol_pred), vol_std)
            plt.show()
        
        return rate
    

if __name__ == "__main__":
    accelerate.utils.set_seed(0)

    train_dataset, val_dataset, test_dataset = utils_experiment.get_combined_image_point_dataset()

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    pred_model = torch.load(f"3dvol_approx/local_stuff/3dglobimgensemble.pth", weights_only=False).to(device)

    if True:
        evaluate(test_loader, pred_model, show_plot=True)
        exit(0)

    #errormodel = 
    
    optimizer = torch.optim.Adam(itertools.chain(model.final.parameters(), model.prec_img.parameters()), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, 0.5)

    best_val = None
    best_model = None
    for epoch in range(30):
        errs_train = []
        model.train()
        for images, points, imagemask, pointmask, label, weight in train_loader:
            images, points, imagemask, pointmask, label, weight = [t.to(device) for t in [images, points, imagemask, pointmask, label, weight]]

            points = utils_3d.to_rigid_invariant_representation(points[:, :, 0:3], points[:, :, 6], num_samples=None)
            volume = model(images, imagemask, points, pointmask)
                
            loss = ((volume.squeeze() - label) ** 2 * weight).mean()
            errs_train.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        err_val = evaluate(val_loader, model)

        if best_val is None or err_val < best_val:
            best_val = err_val
            best_model = copy.deepcopy(model).cpu()

        print(f"epoch {epoch}. Train: {np.mean(errs_train)}")

    torch.save(best_model, f"3dvol_approx/local_stuff/{model_name}")