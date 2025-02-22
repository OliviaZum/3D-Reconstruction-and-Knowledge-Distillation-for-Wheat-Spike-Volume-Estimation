import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import copy
import datasets_3d
import utils_3d
import utils_experiment
from torch.nn import functional as F
import matplotlib.pyplot as plt
import models_3d
import accelerate
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def evaluate(dataloader: DataLoader, model: nn.Module):
    model = model.eval()
    with torch.no_grad():
        losses = []
        for _, batch, _, mask, _  in dataloader:
            batch = batch.to(device)
            mask = mask.to(device)
            batch = utils_3d.to_rigid_invariant_representation(batch[:, :, 0:3], batch[:, :, 6])

            complete_pred, _ = model(batch, mask)
            loss = utils_3d.compare_rigid_invariant_mse(complete_pred, batch)
            losses.append(loss.cpu())
            
        loss = np.mean(losses)
        p97e = np.percentile(losses, 97)
        print(f"Val Loss (hist diff): {loss}")
        print(f"Val loss (variance): {np.var(losses)}")
        print(f"Val loss (p95): {p97e}")
        return p97e

if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset = utils_experiment.get_real_dataset3d(False)
    vloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    if False:
        accelerate.utils.set_seed(0, deterministic=True)
        encoder = models_3d.RigidInvariantPointNet(latent_size=128, output="latent").to(device)
        model = models_3d.RigidInvariantCompletion(encoder, 128).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, 0.5)

        best_val = None
        best_model = None
        for epoch in range(120):
            errs_train = []
            model.train()
            for idx, batch, _, mask, _ in vloader:
                batch = batch.to(device)
                mask = mask.to(device)
                batch = utils_3d.to_rigid_invariant_representation(batch[:, :, 0:3], batch[:, :, 6])
                complete_pred, feature = model(batch, mask)

                loss = utils_3d.compare_rigid_invariant_mse(complete_pred, batch, average=True)

                errs_train.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            err_val = evaluate(val_loader, model)

            if best_val is None or err_val < best_val:
                best_val = err_val
                best_model = copy.deepcopy(model).cpu()

            print(f"epoch {epoch}. Train: {np.mean(errs_train)}")

        torch.save(best_model, "3dvol_approx/local_stuff/autoencoder_traindata.pth")

    if True:
        model = torch.load("3dvol_approx/local_stuff/autoencoder_traindata.pth", weights_only=False).encoder.to(device)
        vol_pred_model = torch.load(f"3dvol_approx/local_stuff/real_volume_model.pth", weights_only=False).to(device)
        vloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        with torch.no_grad():
            features = []
            batches = []
            errors = []
            for _, batch, vol, mask, _  in vloader:
                batches.append(batch.cpu().numpy())
                batch = batch.to(device)
                mask = mask.to(device)
                batch = utils_3d.to_rigid_invariant_representation(batch[:, :, 0:3], batch[:, :, 6])

                feature = model(batch, mask)
                vol_pred = vol_pred_model(batch, mask)

                errors.append(torch.abs(vol_pred.cpu().squeeze() - vol))
                features.append(feature.cpu().numpy())
        
        features = np.concatenate(features)
        batches = np.concatenate(batches)
        errors = np.concatenate(errors)

        plant_id = train_dataset.plant_mapping["plant_id"]

        mds = MDS(n_components=2, dissimilarity="euclidean", random_state=42)
        coords = mds.fit_transform(features)  # Shape: (n, 2)

        fig, ax = plt.subplots()
        p90e = np.percentile(errors, 90)
        print(f"p80 Error: {p90e}")
        errors = ((errors / p90e) + 0.1).clip(0, 1)
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=errors, cmap="Reds")

        def on_click(event):
            if event.inaxes is None:
                return
            cont, ind = sc.contains(event)
            if cont:
                index = ind["ind"][0] 
                sample_name = plant_id.iloc[index]
                show_spike(sample_name, index) 

        def show_spike(name, index):
            points = batches[index]
            print(name)
            utils_3d.visualize_point_cloud_single(points[:, 0:3], points[:, 6])
            
        fig.canvas.mpl_connect("button_press_event", on_click)

        plt.show()


        
