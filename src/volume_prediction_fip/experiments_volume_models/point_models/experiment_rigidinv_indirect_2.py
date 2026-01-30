"""
Uses the ply2volume_model (can be trained with experiment_rigidinv_direct) to train
a RigidInvariantPointNet for volume prediction while using the features predicted by the
ply2volume_model on the corresponding spike ply file as additional loss.
Outperforms direct volume prediction on real data slightly.

Performance on test:
Val MAE: 564.2942504882812
Val Corr: 0.8091378891713066
Steepness: 0.736374898259229
"""

import accelerate
import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import copy
from torch.nn import functional as F
import matplotlib.pyplot as plt
import time

from volume_prediction_fip.experiments_volume_models.shared import utils_experiment, models, datasets, utils_3d
from volume_prediction_fip.utils import helpers

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
arti_2_vol = torch.load(helpers.get_exp_path() / "ply2volume_model.pth", weights_only=False).to(device).eval()
arti_2_vol.output = "features"
# ---- load cached teacher latents ONCE ----
#train_latents = torch.load(
#    helpers.get_exp_path() / "ply2volume_model_bins40_train_latents.pt"
#)
#val_latents = torch.load(
#    helpers.get_exp_path() / "ply2volume_model_bins40_val_latents.pt"
#)
#test_latents = torch.load(
#    helpers.get_exp_path() / "ply2volume_model_bins40_test_latents.pt"
#)

def l2_feature_loss(student, teacher, eps=1e-8):
    # per-sample L2-normalized feature matching
    student = F.normalize(student, dim=1, eps=eps)
    teacher = F.normalize(teacher, dim=1, eps=eps)
    return ((student - teacher) ** 2).sum(dim=1).mean()


def feature_stat_loss(student, teacher):
    # match feature distribution statistics
    mu_s, mu_t = student.mean(dim=0), teacher.mean(dim=0)
    std_s, std_t = student.std(dim=0), teacher.std(dim=0)
    return F.mse_loss(mu_s, mu_t) + F.mse_loss(std_s, std_t)



def evaluate(dataloader: DataLoader, model: nn.Module, latw=5, show_plot=False, MSE_loss=True, L2_loss=False, l2_loss_feature=False, MSE_l2_loss=False):
    model = model.eval()
    with torch.no_grad():
        vol_pred = []
        vol_real = []
        lat_losses = []

        for idx, data_ply, data_depthmap, plymask, mask, vol_batch, _ in dataloader:
            data_ply, data_depthmap, plymask, mask, vol_batch = [t.to(device) for t in (data_ply, data_depthmap, plymask, mask, vol_batch)]
            data_depthmap = utils_3d.to_rigid_invariant_representation(data_depthmap[:, :, 0:3], data_depthmap[:, :, 6], bins=bins)
            data_ply = utils_3d.to_rigid_invariant_representation(data_ply[:, :, 0:3], data_ply[:, :, 6], bins=10)

            ply_latent = arti_2_vol(data_ply, plymask)
            
            #ply_latent = torch.stack([latents[int(i)] for i in idx]).to(device)
            volume, latent = model(data_depthmap, mask)
        
            if MSE_loss: 
                lat_loss = ((ply_latent - latent) ** 2).mean(dim=1)
                lat_losses.append(lat_loss.cpu())

            elif L2_loss:
                lat_loss = ((latent - ply_latent) ** 2).sum(dim=1)
                lat_losses.append(lat_loss.detach().cpu())

            
            elif l2_loss_feature:
                lat_loss = (
                    (F.normalize(latent, dim=1) - F.normalize(ply_latent, dim=1)) ** 2
                ).sum(dim=1)
                lat_losses.append(lat_loss.cpu())
            elif MSE_l2_loss: 
                  # ---- direction (geometry) ----
                student_dir = F.normalize(latent, dim=1)
                teacher_dir = F.normalize(ply_latent, dim=1)

                dir_loss = ((student_dir - teacher_dir) ** 2).sum(dim=1).mean()

                # ---- magnitude (scale), per-sample ----
                norm_loss = (latent.norm(dim=1) - ply_latent.norm(dim=1)) ** 2  # [B]

                # ---- combined KD loss, per-sample ----
                lat_loss =  dir_loss + 0.2 * norm_loss
                lat_losses.append(lat_loss.detach().cpu())

            else:
                raise RuntimeError("Either MSE_loss or l2_loss must be True")
                
            

            vol_pred.append(volume.cpu().squeeze())
            vol_real.append(vol_batch.cpu().squeeze())

        vol_pred = torch.concat(vol_pred)
        vol_real = torch.concat(vol_real)
        lat_losses = torch.concat(lat_losses)

        volloss = ((vol_pred - vol_real) ** 2).mean().item()
        latloss = lat_losses.mean().item()
        loss = volloss + latloss * latw
        print(f"Val MAE: {F.l1_loss(datasets.vol_unorm(vol_pred), datasets.vol_unorm(vol_real))}")
        print(f"Val Corr: {np.corrcoef(vol_real, vol_pred)[0, 1]}")
        A = np.vstack([vol_real, np.ones(len(vol_real))]).T
        linregcoeff, _, _, _ = np.linalg.lstsq(A, vol_pred, rcond=None)
        print(f"Steepness: {linregcoeff[0]}")
        vp = datasets.vol_unorm(vol_pred)
        vr = datasets.vol_unorm(vol_real)
        print(f"Val MAPE: {(torch.abs((vr - vp) / (vr + 1e-8))).mean().item() * 100}")
        print(f"Val Loss (+latent diff): {loss}")
        print(f"Val Loss volume: {volloss}")
        print(f"Val Loss latent: {latloss}")
        if show_plot:
            plt.plot((2000, 8500), (2000, 8500))
            plt.scatter(datasets.vol_unorm(vol_real), datasets.vol_unorm(vol_pred))
            plt.show()
        return {
            "loss": loss,
            "mae": F.l1_loss(datasets.vol_unorm(vol_pred), datasets.vol_unorm(vol_real)).item(),
            "mape": (torch.abs((vr - vp) / (vr + 1e-8))).mean().item() * 100,
            "corr": np.corrcoef(vol_real, vol_pred)[0, 1],
            "steepness": linregcoeff[0],
        }
    
best_val = None
best_epoch = None


if __name__ == "__main__":
    accelerate.utils.set_seed(1, deterministic=True)
    train_dataset, val_dataset, test_dataset = utils_experiment.get_combined_point_real_arti_dataset()
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    ###########################
    bins=10
    MSE_loss = True
    L2_loss = False
    l2_loss_feature = False
    MSE_l2_loss = False

    evaluation = True
    
    ###########################

    model = models.RigidInvariantPointNet(output="volumelatent", bins=bins).to(device)
    model_name = "rigidinv_indirect2.pth"

    
    if evaluation == True:
        
        model = torch.load(helpers.get_exp_path() / f"{model_name}", weights_only=False).to(device)

        # ---- evaluation timing ----
        torch.cuda.synchronize()
        eval_start = time.perf_counter()

        evaluate(test_loader, model, latw=5, show_plot=True, MSE_loss=MSE_loss, L2_loss=L2_loss, l2_loss_feature=l2_loss_feature, MSE_l2_loss=MSE_l2_loss)


        torch.cuda.synchronize()
        eval_time = time.perf_counter() - eval_start

        print(f"Evaluation time: {eval_time:.2f}s")
        exit(0)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.5)
    latw=5

    best_val = None
    best_model = None
    best_metrics = None


    for epoch in range(51):
        torch.cuda.synchronize()
        epoch_start = time.perf_counter()

        errs_train = []
        model.train()
        for idx, data_ply, data_depthmap, plymask, mask, vol_batch, _ in train_loader:
            data_ply, data_depthmap, plymask, mask, vol_batch = [t.to(device) for t in (data_ply, data_depthmap, plymask, mask, vol_batch)]
            
            data_depthmap = utils_3d.to_rigid_invariant_representation(data_depthmap[:, :, 0:3], data_depthmap[:, :, 6], bins=bins)
            data_ply = utils_3d.to_rigid_invariant_representation(data_ply[:, :, 0:3], data_ply[:, :, 6], bins=10)
            
            ply_latent = arti_2_vol(data_ply, plymask)
            #ply_latent = torch.stack([train_latents[int(i)] for i in idx]).to(device)

            volume, latent = model(data_depthmap, mask)
            v_loss = ((volume.squeeze() - vol_batch) ** 2).mean()

            if MSE_loss:
                lat_loss = ((ply_latent - latent) ** 2).mean()
            elif L2_loss:
                lat_loss = ((latent - ply_latent) ** 2).sum(dim=1).mean()

            elif l2_loss_feature: 
                lat_loss = (
                l2_feature_loss(latent, ply_latent)
                + 0.1 * feature_stat_loss(latent, ply_latent))
            elif MSE_l2_loss: 
                  # ---- direction (geometry) ----
                student_dir = F.normalize(latent, dim=1)
                teacher_dir = F.normalize(ply_latent, dim=1)

                dir_loss = ((student_dir - teacher_dir) ** 2).sum(dim=1).mean()

                # ---- magnitude (scale / volume cue) ----
                norm_loss = F.mse_loss(
                    latent.norm(dim=1),
                    ply_latent.norm(dim=1)
                )

                # ---- combined KD loss ----
                lat_loss =  dir_loss + 0.2 * norm_loss

            else:
                raise RuntimeError("Either MSE_loss or l2_loss must be True")
                
            loss = v_loss + latw * lat_loss

            errs_train.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()


        #err_val = evaluate(val_loader, model, latw=latw, MSE_loss=MSE_loss, l2_loss_feature=l2_loss_feature, MSE_l2_loss=MSE_l2_loss)

        val_metrics = evaluate(val_loader, model, latw=latw, MSE_loss=MSE_loss, L2_loss=L2_loss, l2_loss_feature=l2_loss_feature, MSE_l2_loss=MSE_l2_loss)
        err_val = val_metrics["loss"]



        if best_val is None or err_val < best_val:
            best_val = err_val
            best_metrics = val_metrics
            best_model = copy.deepcopy(model).cpu()
            best_epoch = epoch

        # Not terribly clean, but training seems to stabilize towards the end resulting in higher test performance,
        #  but somewhat lower val performance
        #best_model = copy.deepcopy(model).cpu()

        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start

        print(
            f"epoch {epoch:03d} | "
            f"train loss: {np.mean(errs_train):.4f} | "
            f"time: {epoch_time:.2f}s"
        )
        print(f"epoch {epoch}. Train: {np.mean(errs_train)}")
    
    torch.save(best_model, helpers.get_exp_path() / f"{model_name}")

    print("\nBest validation model:")
    print(f"  epoch: {best_epoch}")
    print(f"  val loss: {best_val:.4f}")
    print(f"  val MAE: {best_metrics['mae']:.2f}")
    print(f"  val MAPE: {best_metrics['mape']:.2f}")
    print(f"  val Corr: {best_metrics['corr']:.4f}")
    print(f"  Steepness: {best_metrics['steepness']:.4f}")


#save the best model according to lowest validation loss: with 40 / 10 bins
#use both, teacher and student
#l2 + 0.2 mse loss
#Best validation model:
#  epoch: 29
#  val loss: 10.8620
#  val MAE: 650.64
#  val MAPE: 14.94
#  val Corr: 0.7197
#  Steepness: 0.5466


#save the best model according to lowest validation loss: with 20 / 10 bins
#use both, teacher and student
#l2 + 0.2 mse loss

#Best validation model:
#  epoch: 21
#  val loss: 8.2803
#  val MAE: 644.58
#  val MAPE: 14.59
#  val Corr: 0.7370
#  Steepness: 0.5642


#save the best model according to lowest validation loss: with 30 / 10 bins

#simple l2 loss: 
#Best validation model:
#  epoch: 37
#  val MAE: 627.59
#  val loss: 38.4817
#  val MAPE: 14.50
#  val Corr: 0.7448
# Steepness: 0.5661


#use both, teacher and student
#l2 + 0.2 mse loss

#Best validation model:
#  epoch: 18
#  val loss: 7.4126
#  val MAE: 626.89
#  val MAPE: 14.25
#  val Corr: 0.7451
#  Steepness: 0.5824

#with test loss: 
#Val MAE: 576.268310546875
#Val Corr: 0.8076665608962648
#Steepness: 0.6537299259012452
#Val MAPE: 12.845297157764435
#Val Loss (+latent diff): 7.043581068515778
#Val Loss volume: 0.5299956202507019
#Val Loss latent: 1.3027170896530151
#Evaluation time: 0.72s

#l2 + 0.3 mse loss
#Best validation model:
#  epoch: 24
#  val loss: 8.8888
#  val MAE: 631.32
#  val MAPE: 14.64
#  Steepness: 0.5660
##  val Corr: 0.7341

# lat_loss = 0.2 * dir_loss + norm_loss

#Best validation model:
#  epoch: 13
#  val MAE: 723.86
#  val MAPE: 17.36
#  val loss: 16.3613
#  val Corr: 0.6428
#  Steepness: 0.4638


#use both, teacher and student
#mse loss

#20 bins: 
#Best validation model:
#  epoch: 9
#  val loss: 0.9259

#30 bins: 
#Best validation model:
#  epoch: 9
#  val loss: 0.9166
#  val MAE: 641.88
#  val MAPE: 14.61
#  val Corr: 0.7538
#  Steepness: 0.5898

#test
#Val MAE: 574.1905517578125
#Val Corr: 0.815046263637931
#Steepness: 0.6515581909969232
#Val Loss (+latent diff): 0.8123785257339478
#Val MAPE: 12.96161264181137
#Val Loss volume: 0.5108129978179932
#Val Loss latent: 0.06031310558319092
#Evaluation time: 0.70s


#40 bins: 
#Best validation model:
#  epoch: 9
#Best validation model:
#  val loss: 1.0249

#save teacher, train student only: 

#20 bins
#Best validation model:
#  epoch: 9
#  val loss: 0.9957

#30 bins
#Best validation model:
#  epoch: 18
#  val loss: 0.9577

#test
#Val MAE: 587.7915649414062
#Val Corr: 0.8112999565671054
#Steepness: 0.6648741998821965
#Val MAPE: 13.15186321735382
#Val Loss (+latent diff): 0.8540158718824387
#Val Loss volume: 0.5242446064949036
#Val Loss latent: 0.06595425307750702
#Evaluation time: 0.48s

#40 bins
#Best validation model:
#  epoch: 18
#  val loss: 1.0035



#save the last model: val loss gets worse and worse! 

#val losses: 20 bins
#Val MAE: 618.3377685546875
#Val Corr: 0.7532929932128261
#Steepness: 0.6491418506329394
#Val MAPE: 13.4866863489151
#epoch 050 | train loss: 0.6832 | time: 1.92s
#epoch 50. Train: 0.6832331775128841

#val losses: 30 bins 
#Val MAE: 650.9677124023438
#Val Corr: 0.7282014446347689
#Steepness: 0.5933261962271488
#Val MAPE: 14.595042169094086
#epoch 050 | train loss: 0.6704 | time: 2.25s
#epoch 50. Train: 0.6704428458213806

#val losses: 40 bins 
#Val MAE: 672.867431640625
#Val Corr: 0.7123771836147135
#Steepness: 0.6041701734700448
#Val MAPE: 15.072327852249146
#Val Loss (+latent diff): 1.1912103816866875
#Val Loss volume: 0.7169104814529419
#Val Loss latent: 0.09485998004674911
#epoch 050 | train loss: 0.7665 | time: 2.49s
#epoch 50. Train: 0.7664834871888161


#test losses! 

#ply: 10 bins, field point clouds: 10 bins: - without knowledge distillation
#Val MAE: 582.3629150390625
#Val Corr: 0.8081019165673508
#Steepness: 0.7167405567402324
#Val MAPE: 13.131582736968994
#Evaluation time: 0.56s

#ply: 20 bins, field point clouds: 10 bins: 
#Val MAE: 576.6650390625
#Val Corr: 0.8187178472418254
#Steepness: 0.6893610381022509
#Val MAPE: 12.961918115615845

#ply: 30 bins, field point clouds: 10 bins: 
#Val MAE: 563.712646484375
#Val Corr: 0.8210029692057536
#Steepness: 0.6692366134793574
#Val MAPE: 12.90588229894638

#ply: 40 bins, field point clouds: 10 bins: 
#Val MAE: 572.4437255859375
#Val Corr: 0.8174476173197617
#Steepness: 0.6857174632878631
#Val MAPE: 12.785327434539795

#ply: 50 bins, field point clouds: 10 bins: 
#Val MAE: 584.931396484375
#Val Corr: 0.8068505290691295
#Steepness: 0.6693178274338043
#Val MAPE: 13.37079256772995
#Evaluation time: 0.83s



#ply: 40 bins, field point clouds: 20 bins: 
#Val MAE: 566.8486938476562
#Val Corr: 0.8164168785322037
#Steepness: 0.6896497231590011
#Val MAPE: 13.04566115140915
#Evaluation time: 0.84s




#ply: 50 bins, field point clouds: 20 bins: 
#Val MAE: 565.39892578125
#Val Corr: 0.822429286238175
#Steepness: 0.7113887824637196
#Val MAPE: 12.753893435001373
#Evaluation time: 0.89s