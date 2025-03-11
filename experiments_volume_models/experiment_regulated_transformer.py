import experiments.shared.utils_experiment as utils_experiment
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import copy
import experiments.shared.datasets_3d as datasets_3d
import accelerate
import experiments.shared.models_3d as models_3d
import experiments.shared.utils_experiment as utils_experiment
from torch.nn import functional as F
import matplotlib.pyplot as plt
import pandas as pd
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def evaluate(dataloader: DataLoader, model: nn.Module, show_plot = False, should_print=True, ignore_outliers = False):
    model = model.eval()
    with torch.no_grad():
        vol_pred = []
        vol_real = []
        logvars = []
        weights = []
        for images, plant, imagemask, label, weight in dataloader:
            images = images.to(device)
            imagemask = imagemask.to(device)
            
            volume, logvar = model(images, imagemask)
            volume = volume.cpu().squeeze()
            vol_pred.append(volume)
            vol_real.append(label)
            weights.append(weight)
            logvars.append(logvar.cpu())
        vol_pred = torch.concat(vol_pred)
        vol_real = torch.concat(vol_real)
        weights = torch.concat(weights)
        logvars = torch.concat(logvars)
        if ignore_outliers:
             e = torch.abs(vol_pred - vol_real)
             p9 = torch.quantile(e, 0.9)
             mask = e < p9
             vol_pred, vol_real, weights, logvars = [a[mask] for a in [vol_pred, vol_real, weights, logvars]]

        l = {}
        l["MAE"] = F.l1_loss(datasets_3d.vol_unorm(vol_pred), datasets_3d.vol_unorm(vol_real)).item()
        l["Correlation"] = np.corrcoef(vol_real, vol_pred)[0, 1]
        l["Loss"] = ((vol_pred - vol_real) ** 2 * weights).mean().item()
        A = np.vstack([vol_real, np.ones(len(vol_real))]).T
        linregcoeff, _, _, _ = np.linalg.lstsq(A, vol_pred, rcond=None)
        l["Steepness"] = linregcoeff[0]
        if should_print:
             print(l)
        if show_plot:
            plt.plot((2000, 8500), (2000, 8500))
            plt.plot((2000, 8500), (3000, 9500), c="red")
            plt.plot((2000, 8500), (1000, 7500), c="red")
            plt.scatter(datasets_3d.vol_unorm(vol_real), datasets_3d.vol_unorm(vol_pred))
            plt.show()
            
            """
            plt.scatter(torch.abs(vol_real - vol_pred), torch.exp(logvars))
            plt.show()
            """
        
        return l

def warmup_multiplicative_schedule(e):
     return 0.01 if e < 40 else min(1/ (e * 0.03), 0.1)
        
class SingleMlpMseLoss(nn.Module):
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

def is_better_model(stats, best_stats):
    corr_improve = stats["Correlation"] - best_stats["Correlation"]
    mae_improve =  best_stats["MAE"] - stats["MAE"]
    steep_improve = abs(1 - best_stats["Steepness"]) - abs(1 - stats["Steepness"])
    rate = corr_improve * (50 / 0.03) + mae_improve + steep_improve * (50 / 0.1)

    return rate > 0

if __name__ == "__main__":
    accelerate.utils.set_seed(0)
    #model_name = "image_model.pth"
    #model_name = "fiplike_artifical_image_model.pth"
    model_name = "self-distill-regulated_transformer.pth"

    if model_name == "image_model.pth" or model_name == "self-distill-regulated_transformer.pth":
        train_dataset, val_dataset, test_dataset = utils_experiment.get_image_dataset()
    elif model_name == "fiplike_artifical_image_model.pth":
         _, _, _, train_dataset, val_dataset, test_dataset = utils_experiment.get_artifical_fiplike_dataset()
    else:
         assert False

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, collate_fn=datasets_3d.img_validation_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, collate_fn=datasets_3d.img_validation_collate_fn)

    if True:
        model = torch.load(f"3dvol_approx/local_stuff/{model_name}", weights_only=False).to(device).eval()
        evaluate(test_loader, model, show_plot=True, ignore_outliers=False)
        exit(0)

    model = models_3d.SingleMlp().to(device)
    lossfn = SingleMlpMseLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_multiplicative_schedule)

    best_val = None
    best_model = None
    for epoch in range(400):
        errs_train = []
        model.train()
        for images, label, imagemask, weight in train_loader:
            images, label, imagemask, weight = [t.to(device) for t in [images, label, imagemask, weight]]

            x = model(images, imagemask)
                
            loss = lossfn(x, label, weight)
            errs_train.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        err_val = evaluate(val_loader, model)

        if best_val is None or is_better_model(err_val, best_val):
            best_val = err_val
            best_model = copy.deepcopy(model).cpu()
            best_epoch = epoch

        print(f"epoch {epoch}. Train: {np.mean(errs_train)}")

    print(f"\n Best stats {best_val}")
    torch.save(best_model, f"3dvol_approx/local_stuff/{model_name}")