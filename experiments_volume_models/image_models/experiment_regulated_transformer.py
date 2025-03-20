"""
Train & test for the regulated Transformer model (best model on images)

Ensemble distilled: {'MAE': 549.0514526367188, 'Correlation': 0.829801406399814, 'Loss': 0.11964598298072815, 'Steepness': 0.8014374017067863}
Automatic pairing ensemble distilled: {'MAE': 581.4852905273438, 'Correlation': 0.8111358388770804, 'Loss': 0.13580958545207977, 'Steepness': 0.8031406228252418}
Automatic pairing ensemble distilled (val): {'MAE': 514.7050170898438, 'Correlation': 0.8576632378374552, 'Loss': 0.133570596575737, 'Steepness': 0.9327370908449583}
Automatic pairing ensemble distilled (test, ignore-outliers): {'MAE': 516.2927856445312, 'Correlation': 0.8585407708659728, 'Loss': 0.10024034976959229, 'Steepness': 0.834608569472429}
Direct train: {'MAE': 614.16357421875, 'Correlation': 0.7874227766422761, 'Loss': 0.15318933129310608, 'Steepness': 0.7631462760571315}
Direct train wo distance: {'MAE': 641.6639404296875, 'Correlation': 0.7548744936452643, 'Loss': 0.19171516597270966, 'Steepness': 0.6596010247852248}
Direct train wo distance and seg: {'MAE': 772.6060180664062, 'Correlation': 0.688045394870238, 'Loss': 0.20374730229377747, 'Steepness': 0.7044337368844835}
"""

import experiments_volume_models.shared.utils_experiment as utils_experiment
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import copy
import accelerate
from experiments_volume_models.shared import datasets
from experiments_volume_models.shared import models
import experiments_volume_models.shared.utils_experiment as utils_experiment
from torch.nn import functional as F
import matplotlib.pyplot as plt
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
             p9 = torch.quantile(e, 0.95)
             mask = e < p9
             vol_pred, vol_real, weights, logvars = [a[mask] for a in [vol_pred, vol_real, weights, logvars]]

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
            plt.plot((2000, 8500), (2000, 8500), c="green", label="Perfect prediction")
            plt.plot((2000, 8500), (3000, 9500), c="red", label="Error of 1000 [$mm^3$]")
            plt.plot((2000, 8500), (1000, 7500), c="red")
            plt.scatter(datasets.vol_unorm(vol_real), datasets.vol_unorm(vol_pred))
            plt.xlabel(r"True volume [$mm^3$]")
            plt.ylabel(r"Predicted volume [$mm^3$]")
            plt.legend()
            plt.show()
            
            """
            plt.scatter(torch.abs(vol_real - vol_pred), torch.exp(logvars))
            plt.show()
            """
        
        return l

def warmup_multiplicative_schedule(e):
     return 0.01 if e < 40 else min(1/ (e * 0.03), 0.1)
    

def is_better_model(stats, best_stats):
    corr_improve = stats["Correlation"] - best_stats["Correlation"]
    mae_improve =  best_stats["MAE"] - stats["MAE"]
    steep_improve = abs(1 - best_stats["Steepness"]) - abs(1 - stats["Steepness"])
    rate = corr_improve * (50 / 0.03) + mae_improve + steep_improve * (50 / 0.1)

    return rate > 0

if __name__ == "__main__":
    accelerate.utils.set_seed(0)
    #model_name = "regulatedtransformer-direct.pth"
    #model_name = "fiplike_artifical_image_model.pth"
    model_name = "self-distill-regulated_transformer.pth"
    #model_name = "nodistance-regulated_transformer.pth"
    #model_name = "nodistancenoseg-regulated_transformer.pth"
    use_auto_pair = True

    if model_name == "regulatedtransformer-direct.pth" or model_name == "self-distill-regulated_transformer.pth":
        if use_auto_pair:
            train_dataset, val_dataset, test_dataset = utils_experiment.get_auto_split_dataset()
        else:
            train_dataset, val_dataset, test_dataset = utils_experiment.get_default_image_dataset()
    elif model_name == "fiplike_artifical_image_model.pth":
         _, _, _, train_dataset, val_dataset, test_dataset = utils_experiment.get_artifical_fiplike_dataset()
    elif model_name == "nodistance-regulated_transformer.pth":
        train_dataset, val_dataset, test_dataset = utils_experiment.get_default_image_dataset_wo_distance()
    elif model_name == "nodistancenoseg-regulated_transformer.pth":
        train_dataset, val_dataset, test_dataset = utils_experiment.get_default_image_dataset_wo_distance_and_seg()
    else:
         assert False

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, collate_fn=datasets.img_validation_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, collate_fn=datasets.img_validation_collate_fn)

    if True:
        model = torch.load(f"experiments_volume_models/local_stuff/{model_name}", weights_only=False).to(device).eval()
        evaluate(test_loader, model, show_plot=True, ignore_outliers=False)
        exit(0)

    model = models.RegulatedTransformer().to(device)
    lossfn = models.RegulatedTransformerLoss()
    
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
    torch.save(best_model, f"experiments_volume_models/local_stuff/{model_name}")