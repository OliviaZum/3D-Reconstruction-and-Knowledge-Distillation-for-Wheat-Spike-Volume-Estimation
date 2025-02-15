import torch
from torch.utils.data import DataLoader, Subset
from torch import nn
import numpy as np
import copy
import datasets_3d
import utils_3d
import models_3d
import experiment_utils
from torch.nn import functional as F
import matplotlib.pyplot as plt

"""
Results:
    With cross validated error predictions: 
        Correlation (not a great measure here) essentially 0. 

    With training to directly predict absolute error:
        Correlation ~0.2. At least visually it seems to be able to predict high > 1000 errors to some extent
        (but its pretty bad)

    With normalloss:
        Negatively effects correlation (a bit) and steepness. Error correlation still around 0.2. (From scatter
        plot results are probably even a bit worse than simply predicting the expected error)
"""

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_cross_val_loaders(dataset, split_id, folds, batch_size=16):
    dataset_size = len(dataset)
    fold_size = dataset_size // folds
    
    val_start = split_id * fold_size
    val_end = val_start + fold_size if split_id < folds - 1 else dataset_size
    
    val_indices = list(range(val_start, val_end))
    train_indices = list(range(0, val_start)) + list(range(val_end, dataset_size))
    
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def evaluate(dataloader: DataLoader, model: nn.Module, should_print=True, show_plot = False):
    model = model.eval()
    with torch.no_grad():
        vol_pred = []
        vol_real = []
        for _, batch, vol_batch in dataloader:
            batch = batch.to(device)
            r = utils_3d.to_rigid_invariant_representation(batch[:, :, 0:3], batch[:, :, 6], num_samples=None)
            volume, _ = model(r)
            volume = volume.cpu().squeeze()
            vol_pred.append(volume)
            vol_real.append(vol_batch)
        vol_pred = torch.concat(vol_pred)
        vol_real = torch.concat(vol_real)
        has_conf = len(vol_pred.shape) == 2
        if has_conf:
            vol_err = vol_pred[:, 1]
            vol_pred = vol_pred[:, 0]

        l = {}
        l["MAE"] = F.l1_loss(datasets_3d.DepthMapDataset.vol_unorm(vol_pred), datasets_3d.DepthMapDataset.vol_unorm(vol_real)).item()
        l["Correlation"] = np.corrcoef(vol_real, vol_pred)[0, 1]
        if has_conf:
            l["ErrCorr"] = np.corrcoef(torch.abs(vol_pred - vol_real), vol_err)[0, 1]
        l["Loss"] = F.mse_loss(vol_pred, vol_real).item()
        A = np.vstack([vol_real, np.ones(len(vol_real))]).T
        linregcoeff, _, _, _ = np.linalg.lstsq(A, vol_pred, rcond=None)
        l["Steepness"] = linregcoeff[0]
        if should_print:
            print(l)
        if show_plot:
            plt.plot((2000, 8500), (2000, 8500))
            plt.scatter(datasets_3d.DepthMapDataset.vol_unorm(vol_real), datasets_3d.DepthMapDataset.vol_unorm(vol_pred))
            plt.show()
            if has_conf:
                plt.scatter(torch.abs(vol_pred - vol_real), vol_err)
                plt.xlabel("True error")
                plt.ylabel("Predicted error")
                plt.show()
        
        return l, torch.abs(vol_pred - vol_real)
    
def is_better_model(stats, best_stats):
    corr_improve = stats["Correlation"] - best_stats["Correlation"]
    mae_improve =  best_stats["MAE"] - stats["MAE"]
    steep_improve = abs(1 - best_stats["Steepness"]) - abs(1 - stats["Steepness"])
    rate = corr_improve * (50 / 0.03) + mae_improve + steep_improve * (70 / 0.1)
    if "ErrCorr" in stats:
        err_corr_improve = stats["ErrCorr"] - best_stats["ErrCorr"]
        rate += err_corr_improve * (30 / 0.03)
    return rate > 0 or torch.isnan(torch.tensor(rate))
    
def train_run(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)

    best_val = None
    best_model = None
    best_epoch = None
    for epoch in range(epochs):
        errs_train = []
        model.train()
        for _, batch, vol_batch in train_loader:
            vol_batch = vol_batch.to(device)
            batch = batch.to(device)
            r = utils_3d.to_rigid_invariant_representation(batch[:, :, 0:3], batch[:, :, 6], num_samples=None)

            volume, _ = model(r)
                
            if volume.shape[1] == 2:
                """
                Direct expected error

                loss = ((volume - vol_batch) ** 2).mean()
                err = ((volume[:, 0] - vol_batch[:, 0]) ** 2)
                err_pred_err = (volume[:, 1] - 3 * torch.abs(volume[:, 0] - vol_batch[:, 0])) ** 2
                loss = err.mean() + 0.2 * err_pred_err.mean()
                """
                """
                Normalloss

                prediction = volume[:, 0].squeeze()
                log_var = volume[:, 1].squeeze()
                data_loss = ((prediction - vol_batch[:, 0].squeeze()) ** 2) / (2 * torch.exp(log_var))
                confidence_loss = 0.5 * log_var
                loss = (data_loss + confidence_loss).mean()
                """
                """
                Weighted expected error (overestimation is punished less than underestimation)

                err = torch.abs(volume[:, 0].squeeze() - vol_batch[:, 0].squeeze())
                e = (err - volume[:, 1].squeeze()) ** 2
                mask = e < 0
                e[mask] = e[mask] * 0.3
                e[~mask] = e[~mask]
                loss = torch.mean((volume[:, 0].squeeze() - vol_batch[:, 0].squeeze()) ** 2) + 0.2 * e.mean()
                """
            else:
                loss = ((volume.squeeze() - vol_batch) ** 2).mean()
            errs_train.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        err_val, _ = evaluate(val_loader, model, show_plot=True)

        if best_val is None or is_better_model(err_val, best_val):
            best_val = err_val
            best_model = copy.deepcopy(model).cpu()
            best_epoch = epoch

        print(f"epoch {epoch}. Train: {np.mean(errs_train)}")

    print(f"Taken model from {best_epoch} with {best_val}")
    
    return best_model

class CombinedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pvol = models_3d.RigidInvariantPointNet()
        self.perr = models_3d.RigidInvariantPointNet()

    def forward(self, x):
        v, _ = self.pvol(x)
        e, _ = self.perr(x)

        return torch.concat((v, e), dim=1), None


if __name__ == "__main__":
    model_name = "real_volume_model_confidence.pth" # real_volume
    train_dataset, val_dataset = experiment_utils.get_real_dataset()
    original_val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


    if False:
        model = torch.load(f"3dvol_approx/local_stuff/{model_name}", weights_only=False).to(device).eval()
        evaluate(original_val_loader, model, show_plot=True)
        exit(0)

    if False:
        nfolds = 6

        pred_errors = []
        for fold in range(nfolds):
            tds_split, vds_split = get_cross_val_loaders(train_dataset, fold, nfolds)
            model = models_3d.RigidInvariantPointNet(bins=10).to(device)
            model = train_run(model, tds_split, original_val_loader, 30).to(device)
            h, pred_error = evaluate(vds_split, model, should_print=False)
            print(f"\n\nOn Split {fold}: {h}\n\n")
            pred_errors.append(pred_error)
        
        pred_errors = torch.concat(pred_errors)
        torch.save(pred_errors, "3dvol_approx/local_stuff/predicted_errors_train_fold4.pth")
    else:
        pred_errors = torch.load("3dvol_approx/local_stuff/predicted_errors_train_fold4.pth", weights_only=True)

    train_dataset.set_errors(pred_errors)
    train_loader = DataLoader(train_dataset, 16, True)
    
    combi_model = CombinedModel().to(device)

    train_run(combi_model, train_loader, original_val_loader, 30)
    evaluate(original_val_loader, combi_model, show_plot=True)






