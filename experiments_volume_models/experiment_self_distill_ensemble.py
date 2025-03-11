from typing import Literal
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
from collections.abc import Iterable

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
arti_2_vol = torch.load("3dvol_approx/local_stuff/ply2volume_model.pth", weights_only=False).to(device).eval()
arti_2_vol.output = "features"

class Mixedloader(Iterable):
    def __init__(self, *dataloaders):
        super().__init__()
        self.dataloaders = dataloaders
        self.reset()
        self.prob_load = np.array([len(x) for x in dataloaders], dtype=np.float64)
        self.prob_load /= self.prob_load.sum()
        self.choices = np.arange(len(dataloaders))

    def reset(self):
        self.iterators = [iter(loader) for loader in self.dataloaders]
    
    def __iter__(self):
        while True:
            sel_chain = np.random.choice(self.choices, len(self.choices), False, self.prob_load)
            hasdata = False
            for i in sel_chain:
                try:
                    d = next(self.iterators[i])
                    hasdata = True
                    break
                except StopIteration:
                    continue
            if hasdata:
                yield i, d
            else:
                self.reset()
                break
        
def stats(vol_pred, vol_real, weights = None, show_plot = False):
    if not weights:
        weights = torch.ones_like(vol_pred)
    mae = F.l1_loss(datasets_3d.vol_unorm(vol_pred), datasets_3d.vol_unorm(vol_real))
    print(f"Val MAE: {mae}")
    corr = np.corrcoef(vol_real, vol_pred)[0, 1]
    print(f"Val Corr: {corr}")
    loss = ((vol_pred - vol_real) ** 2 * weights).mean().item()
    print(f"Val Loss: {loss}")
    A = np.vstack([vol_real, np.ones(len(vol_real))]).T
    linregcoeff, _, _, _ = np.linalg.lstsq(A, vol_pred, rcond=None)
    print(f"Steepness: {linregcoeff[0]}")
    rate = -(- mae + corr * (50 / 0.03) - abs(1 - linregcoeff[0]) * (50 / 0.1))
    print(f"Rate: {rate}")
    if show_plot:
        plt.plot((2000, 8500), (2000, 8500))
        plt.scatter(datasets_3d.vol_unorm(vol_real), datasets_3d.vol_unorm(vol_pred))
        plt.show()
    
    return rate, loss

def inference_combined(model, val_ds):
    model = model.eval()
    with torch.no_grad():
        vol_pred = []
        vol_real = []
        weights = []
        for images, points, imagemask, pointmask, label, weight in val_ds:
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

    return vol_pred, vol_real

def inference_pointnet(model, val_ds):
    old_out = model.output
    model.output = "volumelatent"
    model = model.eval()
    with torch.no_grad():
        vol_pred = []
        vol_real = []
        lat_losses = []

        for _, data_ply, data_depthmap, plymask, mask, vol_batch, _ in val_ds:
            data_ply, data_depthmap, plymask, mask, vol_batch = [t.to(device) for t in (data_ply, data_depthmap, plymask, mask, vol_batch)]
            data_depthmap = utils_3d.to_rigid_invariant_representation(data_depthmap[:, :, 0:3], data_depthmap[:, :, 6])
            data_ply = utils_3d.to_rigid_invariant_representation(data_ply[:, :, 0:3], data_ply[:, :, 6])

            ply_latent = arti_2_vol(data_ply, plymask)
            volume, latent = model(data_depthmap, mask)
        
            lat_loss = ((ply_latent - latent) ** 2)
            lat_losses.append(lat_loss.cpu())

            vol_pred.append(volume.cpu().squeeze())
            vol_real.append(vol_batch.cpu().squeeze())

        vol_pred = torch.concat(vol_pred)
        vol_real = torch.concat(vol_real)
        lat_losses = torch.concat(lat_losses)
        loss = ((vol_pred - vol_real) ** 2).mean().item() + lat_losses.mean() * 5

        model.output = old_out

        return loss, vol_pred, vol_real
    
def inference_images(model, val_ds):
    vol_pred = []
    vol_real = []
    weights = []
    with torch.no_grad():
        model = model.eval()
        for images, plant, imagemask, label, weight in val_ds:
            images = images.to(device)
            imagemask = imagemask.to(device)
            
            volume, _ = model(images, imagemask)
            volume = volume.cpu().squeeze()
            vol_pred.append(volume)
            vol_real.append(label)
            weights.append(weight)
        vol_pred = torch.concat(vol_pred)
        vol_real = torch.concat(vol_real)
        weights = torch.concat(weights)
        return vol_pred, vol_real, weights

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

def retrain_image(unlabeled_images, labeled_train_ds, val_ds, nepoch=600):
    model = models_3d.SingleMlp().to(device)
    loader = Mixedloader(unlabeled_images, labeled_train_ds)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_multiplicative_schedule)
    lossfn = SingleMlpMseLoss()

    best_rate = None
    best_model = None
    for epoch in range(nepoch):
        errs_train_label = []
        errs_train_nolabel = []
        model = model.train()
        for dt, data in loader:
            images, label, imagemask, weight = data
            images, label, imagemask, weight = [t.to(device) for t in [images, label, imagemask, weight]]

            x = model(images, imagemask)
                
            loss = lossfn(x, label, weight)
            if dt == 0:
                loss = loss * 0.08
                errs_train_nolabel.append(loss.item())  
            else:
                errs_train_label.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        vol_pred, vol_real, _ = inference_images(model, val_ds)
        rate, _ = stats(vol_pred, vol_real)

        if best_rate is None or rate < best_rate:
            best_rate = rate
            best_model = copy.deepcopy(model).cpu()
            best_epoch = epoch

        print(f"epoch {epoch}. Train (label): {np.mean(errs_train_label)}, Train (nolabel): {np.mean(errs_train_nolabel)}")

    print(f"Best val {best_rate}, epoch {best_epoch}")
    return best_model, best_rate

def retrain_pointnet(unlabeled_depthmaps, labeled_train_ds, val_ds, nepoch = 40):
    model = models_3d.RigidInvariantPointNet(output="volumelatent").to(device)
    loader = Mixedloader(unlabeled_depthmaps, labeled_train_ds)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.5)

    best_val = None
    for epoch in range(nepoch):
        errs_train_label = []
        errs_train_nolabel = []
        model = model.train()
        for dt, data in loader:
            if dt == 0:
                _, batch, vol_batch, mask, _ = data
                batch, vol_batch, mask = [t.to(device) for t in (batch, vol_batch, mask)]

                batch = utils_3d.to_rigid_invariant_representation(batch[:, :, 0:3], batch[:, :, 6])
                volume, _ = model(batch, mask)
                loss = ((volume.squeeze() - vol_batch) ** 2).mean() * 0.4
                errs_train_nolabel.append(loss.item())
            else:
                _, data_ply, data_depthmap, plymask, mask, vol_batch, _ = data
                data_ply, data_depthmap, plymask, mask, vol_batch = [t.to(device) for t in (data_ply, data_depthmap, plymask, mask, vol_batch)]
            
                data_depthmap = utils_3d.to_rigid_invariant_representation(data_depthmap[:, :, 0:3], data_depthmap[:, :, 6])
                data_ply = utils_3d.to_rigid_invariant_representation(data_ply[:, :, 0:3], data_ply[:, :, 6])
                ply_latent = arti_2_vol(data_ply, plymask)

                volume, latent = model(data_depthmap, mask)
                v_loss = ((volume.squeeze() - vol_batch) ** 2).mean()
                lat_loss = ((ply_latent - latent) ** 2).mean()
                loss = v_loss + lat_loss * 5
                errs_train_label.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        err_val, vol_pred, vol_real = inference_pointnet(model, val_ds)
        print(f"Epoch {epoch} on Pointnet; Val: {err_val}, Train (label): {np.mean(errs_train_label)}, Train (nolabel): {np.mean(errs_train_nolabel)}")
        stats(vol_pred, vol_real)
        if best_val is None or err_val < best_val:
            best_val = err_val
            best_model = copy.deepcopy(model).cpu()

    return best_model, best_val
    
def retrain_combined(pretrained_point, pretrained_img, train_ds, val_ds, nepoch=5):
    model = models_3d.Image3dEnsemble()
    model.img_net.load_state_dict(pretrained_img.state_dict(), strict=False)
    model.point_net.load_state_dict(pretrained_point.state_dict(), strict=False)
    model = model.to(device)

    optimizer = torch.optim.Adam(itertools.chain(model.final.parameters(), model.prec_img.parameters()), lr=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, 0.5)

    best_rate = None
    best_model = None
    for epoch in range(nepoch):
        errs_train = []
        model.train()
        for images, points, imagemask, pointmask, label, weight in train_ds:
            images, points, imagemask, pointmask, label, weight = [t.to(device) for t in [images, points, imagemask, pointmask, label, weight]]

            points = utils_3d.to_rigid_invariant_representation(points[:, :, 0:3], points[:, :, 6], num_samples=None)
            volume = model(images, imagemask, points, pointmask)
                
            loss = ((volume.squeeze() - label) ** 2 * weight).mean()
            errs_train.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        print(f"epoch {epoch}. Train: {np.mean(errs_train)}")
        vol_pred, vol_real = inference_combined(model, val_ds)
        rate, _ = stats(vol_pred, vol_real) # weights seem dangerous

        if best_rate is None or rate < best_rate:
            best_rate = rate
            best_model = copy.deepcopy(model).cpu()

    return best_model, best_rate


if __name__ == "__main__":
    accelerate.utils.set_seed(0, deterministic=True)

    train_dataset, val_dataset, test_dataset = utils_experiment.get_combined_image_point_dataset(include_ply=True)
    unlabeled_depthmaps, _, unlabeled_images, _ = utils_experiment.get_unlabeled_dataset()

    point_train_loader = DataLoader(train_dataset.point_dataset, 8, shuffle=True)
    point_val_loader = DataLoader(val_dataset.point_dataset, 8, shuffle=False)
    image_train_loader = DataLoader(train_dataset.image_dataset, 256, shuffle=True)
    image_val_loader = DataLoader(val_dataset.image_dataset, 256, shuffle=False, collate_fn=datasets_3d.img_validation_collate_fn)
    combined_train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    combined_val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model_name = "self-distill-regulated_transformer.pth"

    if False:
        model = torch.load(f"3dvol_approx/local_stuff/{model_name}", weights_only=False).to(device).eval()
        evaluate(val_loader, model, show_plot=True)
        exit(0)

    pretrained_point = torch.load(f"3dvol_approx/local_stuff/real_volume_model.pth", weights_only=False).to(device)
    pretrained_img = torch.load(f"3dvol_approx/local_stuff/image_model.pth", weights_only=False).to(device)

    best_combined_val = None
    #best_pointnet_val, _, _  = inference_pointnet(pretrained_point, point_val_loader)
    best_pointnet_model = copy.deepcopy(pretrained_point.cpu())
    best_img_model = copy.deepcopy(pretrained_img.cpu())
    vol_pred, vol_real, _ = inference_images(best_img_model.to(device), image_val_loader)
    best_image_rate, _ = stats(vol_pred, vol_real)

    for _ in range(2):
        print("\n\nFitting new combined model\n\n")
        combined_model, combined_val = retrain_combined(best_pointnet_model, best_img_model, combined_train_loader, combined_val_loader, nepoch=30)
        if best_combined_val is None or combined_val < best_combined_val:
            best_combined_val = combined_val
        else:
            print(f"Combined stats did not improve with new model. Aborting.")
            break
        print(f"Best combined val {best_combined_val}")

        unlabeled_images.random_sequence_len = False
        cuids = datasets_3d.Image3dCombidataset(unlabeled_images, unlabeled_depthmaps)
        combined_unlabeled_inference = DataLoader(cuids, 16, shuffle=False)
        vol_pred, _ = inference_combined(combined_model.to(device), combined_unlabeled_inference)
        unlabeled_images.random_sequence_len = True


        print("\n\nRetrain images\n\n")
        unlabeled_images.plant_mapping["volume"] = datasets_3d.vol_unorm(vol_pred.cpu())
        image_unlabeled_loader = DataLoader(unlabeled_images, 256, shuffle=True)
        image_model, image_rate = retrain_image(image_unlabeled_loader, image_train_loader, image_val_loader, 300)
        unlabeled_images.plant_mapping["volume"] = torch.nan
        if image_rate < best_image_rate:
            best_img_model = image_model
        else:
            break

        """
        Retraining pointnet leads to worse or similar results than simply training on labeled data

         print("\n\nRetrain pointnet\n\n")
        unlabeled_depthmaps.plant_mapping["volume"] = datasets_3d.vol_unorm(vol_pred.cpu())
        point_unlabeled_train_loader = DataLoader(unlabeled_depthmaps, 8, shuffle=True)
        depth_model, depth_val = retrain_pointnet(point_unlabeled_train_loader, point_train_loader, point_val_loader)
        unlabeled_depthmaps.plant_mapping["volume"] = torch.nan
        if best_pointnet_val > depth_val:
            best_pointnet_val = depth_val
            best_pointnet_model = copy.deepcopy(depth_model.cpu())
        """
       
    

    torch.save(best_img_model, f"3dvol_approx/local_stuff/{model_name}")