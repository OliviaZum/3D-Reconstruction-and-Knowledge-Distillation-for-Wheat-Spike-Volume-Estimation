import torch
from torch.utils.data import DataLoader, Subset
from torch import nn
import numpy as np
import copy
import datasets_3d
import utils_3d
import models_3d
import utils_experiment
from torch.nn import functional as F
import matplotlib.pyplot as plt
import accelerate

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
        for _, batch, vol_batch, mask, _ in dataloader:
            batch = batch.to(device)
            mask = mask.to(device)
            r = utils_3d.to_rigid_invariant_representation(batch[:, :, 0:3], batch[:, :, 6], num_samples=None)
            volume = model(r, mask)
            volume = volume.cpu().squeeze()
            vol_pred.append(volume)
            vol_real.append(vol_batch)
        vol_pred = torch.concat(vol_pred)
        vol_real = torch.concat(vol_real)
        has_conf = len(vol_pred.shape) == 2
        if has_conf:
            vol_err = torch.argmax(vol_pred[:, 1:], dim=1)
            vol_pred = vol_pred[:, 0]

        l = {}
        l["MAE"] = F.l1_loss(datasets_3d.vol_unorm(vol_pred), datasets_3d.vol_unorm(vol_real)).item()
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
            plt.scatter(datasets_3d.vol_unorm(vol_real), datasets_3d.vol_unorm(vol_pred))
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
        rate += err_corr_improve * (20 / 0.03)
    return rate > 0 or torch.isnan(torch.tensor(rate))

class ImprovedLoss(nn.Module):
    def __init__(self, w_pred=1.0, w_uncertainty=0.5, w_reg=0.5):
        """
        Improved loss function that penalizes excessive uncertainty.
        Args:
        w_pred (float): Weight for prediction accuracy.
        w_uncertainty (float): Weight for uncertainty modeling.
        w_reg (float): Weight for penalizing unnecessary uncertainty.
        """
        super().__init__()
        self.w_pred = w_pred
        self.w_uncertainty = w_uncertainty
        self.w_reg = w_reg # New regularization term

    def forward(self, mu, logvar, target):
        """
        Compute the total loss.
        Args:
        mu (torch.Tensor): Predicted mean.
        var (torch.Tensor): Predicted variance (uncertainty).
        target (torch.Tensor): True values.
        Returns:
        torch.Tensor: Total loss.
        """

        # Mean Squared Error for volume prediction
        pred_loss = (mu - target) ** 2
        var = torch.exp(logvar)

        # Gaussian Negative Log-Likelihood Loss
        nll_loss = (pred_loss / (2 * var)) + 0.5 * logvar

        # **New Regularization Term**: Penalize large `σ²` when `err` is small
        small_error_mask = (pred_loss < 0.7) # If error is small, large uncertainty is penalized
        uncertainty_penalty = small_error_mask * var # Directly penalizing high `σ²`

        # Weighted sum of losses
        total_loss = (
            self.w_pred * pred_loss.mean()
            + self.w_uncertainty * nll_loss.mean()
            + self.w_reg * uncertainty_penalty.mean() # Penalizing unnecessary uncertainty
        )

        return total_loss

def train_run(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)

    best_val = None
    best_model = None
    best_epoch = None
    for epoch in range(epochs):
        errs_train = []
        model.train()
        for _, batch, vol_batch, mask, weight in train_loader:
            vol_batch = vol_batch.to(device)
            batch = batch.to(device)
            mask = mask.to(device)
            r = utils_3d.to_rigid_invariant_representation(batch[:, :, 0:3], batch[:, :, 6], num_samples=None)

            pred = model(r, mask)

            merror = (pred[:, 0] - vol_batch) ** 2
            with torch.no_grad():
                mask_large = merror > 1.2
            c_loss = F.cross_entropy(pred[:, 1:], mask_large.long(), weight=torch.tensor([1.0, 8]).to(device)) 
            loss = merror.mean() * 1 + c_loss * 0.3

            errs_train.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        err_val, _ = evaluate(val_loader, model, show_plot=False)

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
        self.perr = models_3d.RigidInvariantPointNet(output="latent")

        self.bins = nn.Linear(128, 2)

    def forward(self, x, mask):
        v = self.pvol(x, mask)
        e = self.perr(x, mask)

        e = self.bins(e)
        
        return torch.cat((v, e), dim=1)


if __name__ == "__main__":
    accelerate.utils.set_seed(1, deterministic=True)

    model_name = "real_volume_model_confidence.pth" # real_volume
    train_dataset, val_dataset, test_dataset = utils_experiment.get_real_dataset3d()
    original_val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    original_test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)



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
        pass
        #pred_errors = torch.load("3dvol_approx/local_stuff/predicted_errors_train_fold4.pth", weights_only=True)

    train_loader = DataLoader(train_dataset, 16, True)
    
    combi_model = CombinedModel().to(device)

    combi_model = train_run(combi_model, train_loader, original_val_loader, 20)

    torch.save(combi_model, "3dvol_approx/local_stuff/combierrormodel.pth")
    combi_model = torch.load("3dvol_approx/local_stuff/combierrormodel.pth", weights_only=False).to(device)
    evaluate(original_test_loader, combi_model, show_plot=True)






