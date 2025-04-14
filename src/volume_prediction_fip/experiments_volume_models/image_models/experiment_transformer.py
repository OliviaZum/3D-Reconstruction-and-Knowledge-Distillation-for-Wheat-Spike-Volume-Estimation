"""
Train + test for transformer model. (AttentionBased)

Has very close results to regulatedTransformer, but it is a bit worse:
{'MAE': 648.7704467773438, 'Correlation': 0.7727548824607278, 'Steepness': 0.7711065434068884}
"""

from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import copy
import accelerate
from torch.nn import functional as F
import matplotlib.pyplot as plt
import torch
from volume_prediction_fip.experiments_volume_models.shared import utils_experiment, models, datasets
from volume_prediction_fip.utils import helpers

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def evaluate(dataloader: DataLoader, model: nn.Module, show_plot = False, should_print=True):
    model = model.eval()
    with torch.no_grad():
        vol_pred = []
        vol_real = []
        weights = []
        for images, plant, imagemask, label, weight in dataloader:
            images = images.to(device)
            imagemask = imagemask.to(device)
            
            volume = model(images, imagemask)
            volume = volume.cpu().squeeze()
            vol_pred.append(volume)
            vol_real.append(label)
            weights.append(weight)
        vol_pred = torch.concat(vol_pred)
        vol_real = torch.concat(vol_real)
        weights = torch.concat(weights)

        l = {}
        l["MAE"] = F.l1_loss(datasets.vol_unorm(vol_pred), datasets.vol_unorm(vol_real)).item()
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
            plt.scatter(datasets.vol_unorm(vol_real), datasets.vol_unorm(vol_pred))
            plt.show()
        
        return l

def warmup_multiplicative_schedule(e):
     return 0.001 if e < 40 else 1/ (e * 0.05)

def is_better_model(stats, best_stats):
    corr_improve = stats["Correlation"] - best_stats["Correlation"]
    mae_improve =  best_stats["MAE"] - stats["MAE"]
    steep_improve = abs(1 - best_stats["Steepness"]) - abs(1 - stats["Steepness"])
    rate = corr_improve * (50 / 0.03) + mae_improve + steep_improve * (50 / 0.1)

    return rate > 0

if __name__ == "__main__":
    accelerate.utils.set_seed(0)
    train_dataset, val_dataset, test_dataset = utils_experiment.get_default_image_dataset()
    train_dataset.min_seq_len = 3
    train_dataset.max_seq_len = 4 # Reduces overfitting

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, collate_fn=datasets.img_validation_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, collate_fn=datasets.img_validation_collate_fn)

    model_name = "transformer.pth"

    if True:
        model = torch.load(helpers.get_exp_path() / f"{model_name}", weights_only=False).to(device).eval()
        evaluate(test_loader, model, show_plot=True)
        exit(0)

    model = models.AttentionBased().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_multiplicative_schedule)

    best_val = None
    best_model = None
    for epoch in range(500):
        errs_train = []
        model.train()
        for images, label, imagemask, weight in train_loader:
            images, label, imagemask, weight = [t.to(device) for t in [images, label, imagemask, weight]]

            x = model(images, imagemask)
                
            loss = ((x.squeeze() - label) ** 2 * weight).mean()
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
    torch.save(best_model, helpers.get_exp_path() / f"{model_name}")