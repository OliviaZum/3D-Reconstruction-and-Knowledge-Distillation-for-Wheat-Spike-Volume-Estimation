"""
Deprecated. Contains experiment to use single image estimates and combine via probability function. 
(With the hope of achieving a more explicit confidence; )
Works quite badly in practice (Actually not horrible, but worse than models which use a learned approach for combination)
"""

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

def max_prob_estimate(means, logvars, weights, mask):
    x_vals = torch.arange(-4, 4.1, 0.1, device=means.device)
    x_vals = x_vals.view(1, 1, -1)  # Shape (1, 1, num_samples)
    
    stds = torch.sqrt(torch.exp(logvars))
    dists = torch.distributions.Normal(means.unsqueeze(-1), stds.unsqueeze(-1))  # Shape (batch, n, num_samples)
    
    probs = torch.exp(dists.log_prob(x_vals))
    probs = probs * (~mask).unsqueeze(-1)

    total_prob = (probs * weights.unsqueeze(-1)).sum(dim=1)
    
    max_indices = total_prob.argmax(dim=-1)
    max_x_vals = x_vals[0, 0, max_indices]
    max_probs = total_prob.gather(1, max_indices.unsqueeze(-1)).squeeze(-1) #?

    return max_x_vals, max_probs


def evaluate(dataloader: DataLoader, model: nn.Module, show_plot = False, should_print=True):
    model = model.eval()
    with torch.no_grad():
        lossfn = SingeImageLoss()
        losses = []
        vol_pred = []
        vol_real = []

        simg_errors = []
        simg_confs = []
        combined_probs = []

        batch_idx = -1
        for images, plant, imagemask, label, weight in dataloader:
            batch_idx += 1
            images = images.to(device)
            imagemask = imagemask.to(device)
            label = label.to(device)
            weight = weight.to(device)
            
            h = model(images, imagemask)
            loss = lossfn(h, label, weight, imagemask).cpu()
            input, conf = h

            if False:
                from scipy.stats import norm
                from pathlib import Path
                import matplotlib.image as mpimg
                base_folder = Path(r"F:\Boxes-ds\segmented_distance_depth")

                for i in range(len(input)):
                    imgs = dataloader.dataset.plant_mapping.loc[batch_idx * len(input) + i, "images"]

                    means_np = input[i, ~imagemask[i]].cpu().numpy()
                    stds_np = torch.sqrt(torch.exp(conf[i, ~imagemask[i]])).cpu().numpy()

                    x = np.linspace(means_np.min() - 3, means_np.max() + 3, 300)

                    img_count = min(len(imgs), 12)  # Max 12 images

                    # --- First Window: Plot Normal Distributions ---
                    plt.figure(figsize=(8, 5))
                    for j in range(len(means_np)):
                        y = norm.pdf(x, means_np[j], stds_np[j])
                        plt.plot(x, y, label=f"Dist {j+1}")

                    plt.axvline(label[i].item(), color='red', linestyle='dashed', label="True Value")
                    plt.xlabel("x")
                    plt.ylabel("Density")
                    plt.title("Visualization of Normal Distributions")
                    plt.ylim(top=1)
                    plt.legend()
                    plt.show(block=False)  # Show first window without blocking execution

                    # --- Second Window: Show Images ---
                    fig, axes = plt.subplots(1, img_count, figsize=(img_count * 2, 2))
                    
                    if img_count == 1:  # `plt.subplots(1, 1)` returns a single axis, not a list
                        axes = [axes]

                    for idx in range(img_count):
                        img_path = base_folder / imgs[idx]
                        img = mpimg.imread(img_path)
                        axes[idx].imshow(img)
                        axes[idx].axis("off")  # No labels, no ticks

                    plt.show()
                
            target = label.unsqueeze(1)
            simg_error = torch.abs(input - target)[~imagemask].cpu()
            simg_conf = conf[~imagemask].cpu()
            simg_errors.append(simg_error)
            simg_confs.append(simg_conf)

            msum = (~imagemask).sum(dim=-1).reshape((-1, 1))
            input[imagemask] = 0
            combined_vol_pred = torch.sum(input, dim=1, keepdim=True) / msum

            var = torch.exp(conf)
            var[imagemask] = 0.3
            var[imagemask] = 1 # Give a bit of variance for non existing images (decreasing confidence)
            mean_sq = torch.sum(var + input**2, dim=1, keepdim=True) / msum
            combined_var = mean_sq - combined_vol_pred**2

            combined_probs.append(combined_var.cpu())
            vol_pred.append(combined_vol_pred.cpu().squeeze())
            vol_real.append(label.cpu())
            losses.append(loss)

        loss = torch.mean(torch.stack(losses))
        vol_pred = torch.concat(vol_pred)
        vol_real = torch.concat(vol_real)
        combined_probs = torch.concat(combined_probs)
        print(f"Val MAE: {F.l1_loss(datasets_3d.vol_unorm(vol_pred), datasets_3d.vol_unorm(vol_real))}")
        print(f"Val Corr: {np.corrcoef(vol_real, vol_pred)[0, 1]}")
        print(f"Val Loss: {loss}")
        A = np.vstack([vol_real, np.ones(len(vol_real))]).T
        linregcoeff, _, _, _ = np.linalg.lstsq(A, vol_pred, rcond=None)
        print(f"Steepness: {linregcoeff[0]}")
        if show_plot:
            """
            simg_errors = torch.concat(simg_errors).numpy()
            simg_confs = torch.concat(simg_confs).numpy()
            plt.scatter(simg_errors, simg_confs)
            plt.show()
            """

            """
            plt.plot((2000, 8500), (2000, 8500))
            plt.scatter(datasets_3d.vol_unorm(vol_real), datasets_3d.vol_unorm(vol_pred))
            plt.show()
            """

            plt.scatter(np.abs(vol_pred - vol_real), combined_probs)
        plt.show()

        return loss

def warmup_multiplicative_schedule(e):
     return 0.01 if e < 10 else min(1/ (e * 0.03), 1)

class SingeImageLoss(nn.Module):
        def __init__(self, weights = (1, 0.5, 0.3, 0.05), *args, **kwargs):
              super().__init__(*args, **kwargs)
              self.weights = weights
              self.is_var_penalty = False

        def forward(self, input: torch.Tensor, target: torch.Tensor, loss_scale: torch.Tensor, mask) -> torch.Tensor:
            input, conf = input
            target = target.unsqueeze(1)
            loss_scale = loss_scale.unsqueeze(1)
            variance = torch.exp(conf)
            err_single_img = ((input - target) ** 2) / (2 * variance) + 0.5 * conf

            if self.is_var_penalty:
                with torch.no_grad():
                    penality_up = variance > 15
                    prob_density = torch.exp(-0.5 * ((input - target) ** 2) / variance) / torch.sqrt(2 * torch.pi * variance)
                    penalty_down = (prob_density < 0.1) & (~penality_up)
                print(f"!! Down: {penalty_down[~mask].sum()}, Up: {penality_up[~mask].sum()}, Mean var {variance[~mask].mean()}")
                err_single_img = err_single_img + penalty_down * torch.exp(- conf) * 20 + penality_up * torch.exp(conf) * 20

            # Loss scaling sicnificantly degrades performence here (probably simply cannot do that in a proper fashion)
            err_single_img = err_single_img * (~mask)# * loss_scale
            return err_single_img.mean()

if __name__ == "__main__":
    accelerate.utils.set_seed(0)
    train_dataset, val_dataset, test_dataset = utils_experiment.get_image_dataset()

    train_dataset.min_seq_len = 12
    train_dataset.max_seq_len = 12 # Always give all images
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, collate_fn=datasets_3d.img_validation_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, collate_fn=datasets_3d.img_validation_collate_fn)

    model_name = "single_image_model.pth"

    if True:
        model = torch.load(f"3dvol_approx/local_stuff/{model_name}", weights_only=False).to(device).eval()
        evaluate(test_loader, model, show_plot=True)
        exit(0)

    model = models_3d.SingleImageModel().to(device)
    lossfn = SingeImageLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_multiplicative_schedule)

    best_val = None
    best_model = None
    for epoch in range(150):
        errs_train = []
        if epoch == 100:
            lossfn.is_var_penalty = True
        model.train()
        for images, label, imagemask, weight in train_loader:
            images, label, imagemask, weight = [t.to(device) for t in [images, label, imagemask, weight]]

            x = model(images, imagemask)
                
            loss = lossfn(x, label, weight, imagemask)
            errs_train.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        err_val = evaluate(val_loader, model)

        if epoch > 130 and (best_val is None or err_val < best_val):
            best_val = err_val
            best_model = copy.deepcopy(model).cpu()
            best_epoch = epoch

        print(f"epoch {epoch}. Train: {np.mean(errs_train)}")

    print(f"\n Best stats {best_val}, epoch {best_epoch}")
    torch.save(best_model, f"3dvol_approx/local_stuff/{model_name}")