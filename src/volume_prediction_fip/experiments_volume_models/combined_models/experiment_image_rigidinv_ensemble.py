from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import copy
from torch.nn import functional as F
import matplotlib.pyplot as plt
import torch
import itertools
import accelerate

from volume_prediction_fip.experiments_volume_models.shared import utils_experiment, models, datasets, utils_3d
from volume_prediction_fip.utils import helpers

"""
An ensemble of the RigidInvariant PointNet (trained with Scan supervision) and the regulated Transformer

Results on test (both models are trained directly):
Val MAE: 521.1376342773438
Val Corr: 0.8488245778778427
Steepness: 0.789510382404692

Results on test (regulated transformer is self distilled):
Val MAE: 531.038330078125
Val Corr: 0.8497957443458352
Steepness: 0.8254965274223702
"""

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def evaluate(dataloader: DataLoader, model: nn.Module, show_plot = False):
    model = model.eval()
    with torch.no_grad():
        vol_pred = []
        vol_real = []
        weights = []
        for images, points, imagemask, pointmask, label, weight in dataloader:
            images, points, imagemask, pointmask = [t.to(device) for t in [images, points, imagemask, pointmask]]
            points = utils_3d.to_rigid_invariant_representation(points[:, :, 0:3], points[:, :, 6], num_samples=None)
            volume = model(images, imagemask, points, pointmask)
            volume = volume.cpu().squeeze()
            vol_pred.append(volume)
            vol_real.append(label)
            weights.append(weight)
        vol_pred = torch.concat(vol_pred)
        vol_real = torch.concat(vol_real)
        weights = torch.concat(weights)

        mae = F.l1_loss(datasets.vol_unorm(vol_pred), datasets.vol_unorm(vol_real))
        print(f"Val MAE: {mae}")
        corr = np.corrcoef(vol_real, vol_pred)[0, 1]
        print(f"Val Corr: {corr}")
        print(f"Val Loss: {((vol_pred - vol_real) ** 2 * weights).mean().item()}")
        A = np.vstack([vol_real, np.ones(len(vol_real))]).T
        linregcoeff, _, _, _ = np.linalg.lstsq(A, vol_pred, rcond=None)
        print(f"Steepness: {linregcoeff[0]}")
        vp = datasets.vol_unorm(vol_pred)
        vr = datasets.vol_unorm(vol_real)
        print(f"Val MAPE: {(torch.abs((vr - vp) / (vr + 1e-8))).mean().item() * 100}")

        rate = -(- mae + corr * (50 / 0.03) - abs(1 - linregcoeff[0]) * (50 / 0.1))
        print(f"Rate: {rate}")
        if show_plot:
            plt.plot((2000, 8500), (2000, 8500))
            plt.scatter(datasets.vol_unorm(vol_real), datasets.vol_unorm(vol_pred))
            plt.show()
        
        return rate

if __name__ == "__main__":
    accelerate.utils.set_seed(0)

    train_dataset, val_dataset, test_dataset = utils_experiment.get_combined_image_point_dataset()

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model_name = "3dglobimgensemble.pth"

    evaluation = True
    if evaluation == True:
        model = torch.load(helpers.get_exp_path() / f"{model_name}", weights_only=False).to(device).eval()
        evaluate(test_loader, model, show_plot=True)
        exit(0)

    model = models.Image3dEnsemble()

    pretrained_point = torch.load(helpers.get_exp_path() / f"rigidinv_indirect2.pth", weights_only=False)
    pretrained_img = torch.load(helpers.get_exp_path() / f"regulatedtransformer-direct.pth", weights_only=False)
    #pretrained_img = torch.load(helpers.get_exp_path() / f"self-distill-regulated_transformer.pth", weights_only=False)[]

    model.img_net.load_state_dict(pretrained_img.state_dict(), strict=False)
    model.point_net.load_state_dict(pretrained_point.state_dict(), strict=False)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(itertools.chain(model.final.parameters(), model.prec_img.parameters()), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)

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

        if best_val is None or (err_val < best_val and epoch > 15):
            best_val = err_val
            best_model = copy.deepcopy(model).cpu()

        print(f"epoch {epoch}. Train: {np.mean(errs_train)}")

    torch.save(best_model, helpers.get_exp_path() / f"{model_name}")