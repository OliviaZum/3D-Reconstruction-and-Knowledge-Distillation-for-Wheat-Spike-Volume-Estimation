"""
Directly training the RigidInvariantPointNet for volume prediction.
By changing the modelname different datasets can be used (the real dataset,
the ply dataset [i.e. points sampled from the 3d scans], or the fiplike artificial
dataset)

Performance on real:
Val MAE: 580.3414916992188
Val Corr: 0.8002600909545133
Steepness: 0.6788825071122576

Performance on ply files:
Val MAE: 141.0824432373047
Val Corr: 0.9881163633508186
Steepness: 0.9930740456518544

Performance on ply files (voxelized):
Val MAE: 108.59272766113281
Val Corr: 0.9923584956654167
Steepness: 1.00016117094416

Fiplike uniform:
Val MAE: 333.9336242675781
Val Corr: 0.9393371638387208
Steepness: 0.9940054408544241

Fiplike normal:
Val MAE: 284.6448059082031
Val Corr: 0.9530460571009036
Steepness: 0.9279888669048981
"""

import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import copy
from torch.nn import functional as F
import matplotlib.pyplot as plt
import accelerate
from volume_prediction_fip.experiments_volume_models.shared import utils_experiment, models, datasets, utils_3d
from volume_prediction_fip.utils import helpers
import time



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def save_latents(dataloader: DataLoader, model: nn.Module, bins: int, out_name: str):
    model = model.eval()
    model.output = "feature"   # IMPORTANT

    latents = {}

    with torch.no_grad():
        for idx, batch, _, mask, _ in dataloader:
            batch = batch.to(device)
            mask = mask.to(device)

            r = utils_3d.to_rigid_invariant_representation(
                batch[:, :, 0:3],
                batch[:, :, 6],
                num_samples=None,
                bins=bins
            )

            latent = model(r, mask)  # [B, latent_dim]

            for i, sample_idx in enumerate(idx):
                latents[int(sample_idx)] = latent[i].cpu()

    out_path = helpers.get_exp_path() / out_name
    torch.save(latents, out_path)
    print(f"Saved {len(latents)} latents to {out_path}")
        

def evaluate(dataloader: DataLoader, model: nn.Module, show_plot = False, bins=10):
    model = model.eval()
    with torch.no_grad():
        vol_pred = []
        vol_real = []
        weights = []
        for _, batch, vol_batch, mask, weight in dataloader:
            batch = batch.to(device)
            mask = mask.to(device)
            r = utils_3d.to_rigid_invariant_representation(batch[:, :, 0:3], batch[:, :, 6], num_samples=None, bins=bins)
            volume = model(r, mask)
            volume = volume.cpu().squeeze()
            vol_pred.append(volume)
            vol_real.append(vol_batch)
            weights.append(weight)
        vol_pred = torch.concat(vol_pred)
        vol_real = torch.concat(vol_real)
        weights = torch.concat(weights)

        loss = ((vol_pred - vol_real) ** 2).mean().item()
        #loss = ((vol_pred - vol_real) ** 2).mean().item()
        print(f"Val MAE: {F.l1_loss(datasets.vol_unorm(vol_pred), datasets.vol_unorm(vol_real))}")
        print(f"Val Corr: {np.corrcoef(vol_real, vol_pred)[0, 1]}")
        print(f"Val Loss: {loss}")
        A = np.vstack([vol_real, np.ones(len(vol_real))]).T
        linregcoeff, _, _, _ = np.linalg.lstsq(A, vol_pred, rcond=None)
        print(f"Steepness: {linregcoeff[0]}")
        vp = datasets.vol_unorm(vol_pred)
        vr = datasets.vol_unorm(vol_real)
        print(f"Val MAPE: {(torch.abs((vr - vp) / (vr + 1e-8))).mean().item() * 100}")
        if show_plot:
            plt.plot((2000, 8500), (2000, 8500))
            plt.scatter(datasets.vol_unorm(vol_real), datasets.vol_unorm(vol_pred))
            plt.show()
        return loss

if __name__ == "__main__":
    # Results vary quite a bit depending on seed
    accelerate.utils.set_seed(1, deterministic=True)
    
    #model_name = "real_volume_model-direct.pth"
    model_name = "ply2volume_model.pth"
    #model_name = "ply2volume_voxelized_model.pth"
    #model_name = "fiplike_uniform_ply_model.pth"
    #model_name = "fiplike_normal_ply_model.pth"

    if model_name == "ply2volume_model.pth":
        train_dataset, val_dataset, test_dataset = utils_experiment.get_ply_dataset(force_recompute=False, voxelize=True)
    elif model_name == "ply2volume_voxelized_model.pth":
        train_dataset, val_dataset, test_dataset = utils_experiment.get_ply_dataset(force_recompute=False, voxelize=True)
    elif model_name == "real_volume_model-direct.pth":
        train_dataset, val_dataset, test_dataset = utils_experiment.get_real_dataset3d(force_recompute=False)
    elif model_name == "fiplike_uniform_ply_model.pth":
        train_dataset, val_dataset, test_dataset = utils_experiment.get_artificial_dataset_fiplike_uniform("depth", force_recompute=False)
    elif model_name == "fiplike_normal_ply_model.pth":
        train_dataset, val_dataset, test_dataset = utils_experiment.get_artificial_dataset_fiplike_normal("depth", force_recompute=False)
    else:
        assert False
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    ##########################
    bins = 10
    evaluation = True
    ##########################



    if evaluation == True:
        model = torch.load(helpers.get_exp_path() / f"{model_name}", weights_only=False).to(device).eval()

         # ---- evaluation timing ----
        torch.cuda.synchronize()
        eval_start = time.perf_counter()

        evaluate(test_loader, model, show_plot=True, bins=bins)

        torch.cuda.synchronize()
        eval_time = time.perf_counter() - eval_start

        print(f"Evaluation time: {eval_time:.2f}s")

        #save_latents(
        #    dataloader=DataLoader(train_dataset, batch_size=8, shuffle=False),
        #    model=model,
        #    out_name=f"{model_name.replace('.pth','')}_bins{bins}_train_latents.pt"
        #    bins=bins,
        #)

        #save_latents(
        #    dataloader=DataLoader(val_dataset, batch_size=8, shuffle=False),
        #    model=model,
        #    bins=bins,
        #    out_name=f"{model_name.replace('.pth','')}_bins{bins}_val_latents.pt"
        #)

        #save_latents(
        #    dataloader=DataLoader(test_dataset, batch_size=8, shuffle=False),
        #    model=model,
        #    bins=bins,
        #    out_name=f"{model_name.replace('.pth','')}_bins{bins}_test_latents.pt"
        #)

        exit(0)

    model = models.RigidInvariantPointNet(bins=bins).to(device)

    """
    Using a pretrained model (does not work better than just direct training; deprecated)

    pretrain_model = torch.load(helpers.get_exp_path() / "pretrained_model.pth", weights_only=False)
    new_state_dict = model.state_dict()
    for k, v in pretrain_model.state_dict().items():
        if "l1" in k:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        [{"params": model.l1.parameters(), "lr": 0.0001},
        {"params": model.lastlin.parameters(), "lr": 0.002}], lr=0.002)
    """


    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.5)

    best_val = None
    best_model = None
    for epoch in range(60):
        torch.cuda.synchronize()
        epoch_start = time.perf_counter()

        errs_train = []
        model.train()

        for idx, batch, vol_batch, mask, weight in train_loader:
            vol_batch = vol_batch.to(device)
            mask = mask.to(device)
            batch = batch.to(device)
            weight = weight.to(device)
            
            r = utils_3d.to_rigid_invariant_representation(batch[:, :, 0:3], batch[:, :, 6], num_samples=None, bins = bins)
            volume = model(r, mask)
                
            loss = ((volume.squeeze() - vol_batch) ** 2).mean()
            errs_train.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        err_val = evaluate(val_loader, model, bins=bins)

        if best_val is None or err_val < best_val:
            best_val = err_val
            best_model = copy.deepcopy(model).cpu()
        
        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start

        print(
            f"epoch {epoch:03d} | "
            f"train loss: {np.mean(errs_train):.4f} | "
            f"time: {epoch_time:.2f}s"
        )

        print(f"epoch {epoch}. Train: {np.mean(errs_train)}")

    torch.save(best_model, helpers.get_exp_path() / f"{model_name}")

#learning rate: 
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.5)
#take best validation model 

#20 bins: 

#Val MAE: 110.01852416992188
#Val Corr: 0.9948335476850508
#Val Loss: 0.021435106173157692
#Steepness: 1.0037353435423442
#Val MAPE: 2.607983537018299
#Evaluation time: 0.48s

#30 bins: 
#Val MAE: 114.49935913085938
#Val Corr: 0.9936629390921988
#Val Loss: 0.02451905608177185
#Steepness: 1.0093361417796258
#Val MAPE: 2.752685733139515
#Evaluation time: 0.55s

#40 bins
#Val MAE: 110.071533203125
#Val Corr: 0.995961092572018
#Val Loss: 0.01948995143175125
#Steepness: 1.0459007099380864
#Val MAPE: 2.5372734293341637
#Evaluation time: 0.61s


#set to 10 bins: 
#Val MAE: 104.51681518554688
#Val Corr: 0.993079275072471
#Val Loss: 0.02155246026813984
#Steepness: 0.9681114811809532
#Val MAPE: 2.290513552725315

#set to 20 bins: 
#Val MAE: 109.15414428710938
#Val Corr: 0.9949240516254767
#Val Loss: 0.020467158406972885
#Steepness: 1.0047228355129814
#Val MAPE: 2.5854114443063736

#set to 30 bins: 
#Val MAE: 113.08631134033203
#Val Corr: 0.9936150969452122
#Val Loss: 0.02444547414779663
#Steepness: 1.009861118974362
#Val MAPE: 2.748478017747402

#set to 40 bins: 
#Val MAE: 109.586669921875
#Val Corr: 0.9958728295554264
#Val Loss: 0.01945229060947895
#Steepness: 1.0454113786749848
#Val MAPE: 2.523437701165676

#set to 50 bins: 
#Val MAE: 105.56632232666016
#Val Corr: 0.9937977754913688
#Steepness: 1.0004258200443747
#Val Loss: 0.019919315353035927
#Val MAPE: 2.4266762658953667
#Evaluation time: 0.69s

#real data set: 
#Val MAE: 594.2637329101562
#Val Corr: 0.7960995003716316
#Val Loss: 0.5577086806297302
#Steepness: 0.6130315211274003
#Val MAPE: 13.38384598493576
#Evaluation time: 0.44s



    
 
                
