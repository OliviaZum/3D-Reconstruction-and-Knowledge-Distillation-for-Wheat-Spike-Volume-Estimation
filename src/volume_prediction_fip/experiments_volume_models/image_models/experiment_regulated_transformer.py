"""
Train & test for the regulated Transformer model (best model on images)

Ensemble distilled: {'MAE': 589.3368530273438, 'Correlation': 0.8251570501247848, 'Steepness': 0.8735967867009864, 'Loss': 0.12570838630199432}
Automatic pairing ensemble distilled: {'MAE': 623.96533203125, 'Correlation': 0.803359489635608, 'Steepness': 0.8695522221001687, 'Loss': 0.14378240704536438}
precise:
    Season stats: {0: {'MAE': 581.8746948242188, 'Correlation': 0.7194981259374609, 'Steepness': 0.8159482130491466},
      1: {'MAE': 663.4834594726562, 'Correlation': 0.7225280516647276, 'Steepness': 0.6450321278652655},
    2: {'MAE': 633.3892822265625, 'Correlation': 0.7182158378345234, 'Steepness': 0.7621034496923911}}
Median mean small 167.3112030029297
Median mean large 384.1881408691406

Automatic pairing ensemble distilled (val): {'MAE': 514.7050170898438, 'Correlation': 0.8576632378374552, 'Loss': 0.133570596575737, 'Steepness': 0.9327370908449583}
Automatic pairing ensemble distilled (test, ignore-outliers): {'MAE': 516.2927856445312, 'Correlation': 0.8585407708659728, 'Loss': 0.10024034976959229, 'Steepness': 0.834608569472429}
Direct train: {'MAE': 614.16357421875, 'Correlation': 0.7874227766422761, 'Loss': 0.15318933129310608, 'Steepness': 0.7631462760571315}
Direct train wo distance: {'MAE': 641.6639404296875, 'Correlation': 0.7548744936452643, 'Loss': 0.19171516597270966, 'Steepness': 0.6596010247852248}
Direct train wo distance and seg: {'MAE': 772.6060180664062, 'Correlation': 0.688045394870238, 'Loss': 0.20374730229377747, 'Steepness': 0.7044337368844835}

Artificial bestpose: {'MAE': 229.02496337890625, 'Correlation': 0.9740162925000551, 'Loss': 0.018728487193584442, 'Steepness': 0.9941033835018213}
Artificial randompose: {'MAE': 247.00242614746094, 'Correlation': 0.970975135459737, 'Loss': 0.026273004710674286, 'Steepness': 1.0532637860866683}
Artificial fiplike normal: {'MAE': 477.668212890625, 'Correlation': 0.8881649636876028, 'Loss': 0.08979550004005432, 'Steepness': 0.9243962589100745
Artificial fiplike uniform: {'MAE': 495.8588562011719, 'Correlation': 0.8653908952381895, 'Loss': 0.08875340968370438, 'Steepness': 0.908395792635748}
"""
#use: export CUBLAS_WORKSPACE_CONFIG=:4096:8
#export CUBLAS_WORKSPACE_CONFIG=:16:8

fine_tuning = False


from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import copy
import accelerate
from torch.nn import functional as F
import matplotlib.pyplot as plt
import torch
import pandas as pd
import warnings
from torch.utils.data import Subset
from volume_prediction_fip.experiments_volume_models.shared import utils_experiment, models, datasets
from volume_prediction_fip.utils import helpers
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap
from sklearn.preprocessing import StandardScaler
import time

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_season(plant_id):
    range, row, p = plant_id.split("_")
    range, row, p = int(range), int(row), int(p)
    if row >= 6 and row <= 10:
        if p in [1, 10, 2]:
            return 0
        elif p in [3, 4]:
            return 1
        else:
            return 2
    elif row >= 15 and row <= 19:
        if p in [1, 2, 3]:
            return 0
        elif p in [4, 5]:
            return 1
        else:
            warnings.warn(f"A sample from the last date of 2024 is present in the evaluation")
            return 2
    else:
        warnings.warn(f"Row that is not in range of 6-10, 15-19 has been found {plant_id}")
        return 7

def get_stats(vol_pred, vol_real):
    l = {}
    l["MAE"] = F.l1_loss(datasets.vol_unorm(vol_pred), datasets.vol_unorm(vol_real)).item()
    l["Correlation"] = np.corrcoef(vol_real, vol_pred)[0, 1]
    A = np.vstack([vol_real, np.ones(len(vol_real))]).T
    linregcoeff, _, _, _ = np.linalg.lstsq(A, vol_pred, rcond=None)
    l["Steepness"] = linregcoeff[0]
    vp = datasets.vol_unorm(vol_pred)
    vr = datasets.vol_unorm(vol_real)
    l["MAPE"] = (torch.abs((vr - vp) / (vr + 1e-8))).mean().item() * 100
    return l

def group_check(vol_real, vol_pred, k, sampling_points, group_by_gt = True):
    medians_pred = []
    medians_real = []

    for sp in sampling_points:
        cmp = vol_real if group_by_gt else vol_pred
        dists = torch.abs(cmp - sp)
        idx = torch.topk(dists, k, largest=False).indices
        medians_pred.append(torch.median(vol_pred[idx]))
        medians_real.append(torch.median(vol_real[idx]))

    return torch.tensor(medians_real), torch.tensor(medians_pred)


def evaluate(dataloader: DataLoader, model: nn.Module, show_plot = False, should_print=True, fraction_include = 1, detail_eval = False, save_test_predictions=None):
    total_infer_time = 0
    total_spikes = 0
    model = model.eval()
    with torch.no_grad():
        vol_pred = []
        vol_real = []
        logvars = []
        weights = []
        plants = []
        for images, plant, imagemask, label, weight in dataloader:
            images = images.to(device)
            imagemask = imagemask.to(device)
            label = label.to(device)
            weight = weight.to(device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()
            
            volume, logvar = model(images, imagemask)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.time()

            total_infer_time += (end - start)
            total_spikes += images.size(0)

            volume = volume.cpu().squeeze()
            vol_pred.append(volume)
            vol_real.append(label)
            weights.append(weight)
            logvars.append(logvar.cpu())
            for p in plant:
                plants.append(p["plant_id"])
        vol_pred = torch.concat(vol_pred).cpu()
        vol_real = torch.concat(vol_real).cpu()
        weights = torch.concat(weights).cpu()
        logvars = torch.concat(logvars).cpu()

        if save_test_predictions is not None: 
            df = pd.DataFrame({
                "plant_id": plants,
                "volume_pred_norm": vol_pred.numpy(),
                "volume_real_norm": vol_real.numpy(),
                "volume_pred_mm3": datasets.vol_unorm(vol_pred).numpy(),
                "volume_real_mm3": datasets.vol_unorm(vol_real).numpy(),
                "logvar": logvars.squeeze().numpy(),
                "weight": weights.numpy()

            })
            df.to_csv(save_test_predictions, index=False)
            print(f"Saved predictions to {save_test_predictions}")
        
        if fraction_include < 1:
            e = torch.abs(vol_pred - vol_real)
            p95 = torch.quantile(e, fraction_include)
            mask = e < p95
            vol_pred, vol_real, weights, logvars = [a[mask] for a in [vol_pred, vol_real, weights, logvars]]
            plants = [p for m, p in zip(mask, plants) if m]

        l = get_stats(vol_pred, vol_real)
        l["Loss"] = ((vol_pred - vol_real) ** 2 * weights).mean().item()

        print(f"Total inference time: {total_infer_time:.4f}s")
        print(f"Inference time per spike: {total_infer_time / total_spikes:.6f}s")

        if should_print:
             print(l)

        if detail_eval:
            seasons = np.array(list(map(get_season, plants)))
            season_stats = {day: get_stats(vol_pred[seasons == day], vol_real[seasons == day]) for day in np.unique(seasons)}
            print(f"Season stats: {season_stats}")
            if show_plot:
                season_to_colors = {
                    0: [0, 1, 1],
                    1: [0, 1, 0],
                    2: [0, 0, 1],
                    7: [1, 0, 1]
                }
                season_to_label = {
                    0: "Growth stage 0",
                    1: "Growth stage 1",
                    2: "Growth stage 2"
                }

                k1, k2 = group_check(datasets.vol_unorm(vol_real), datasets.vol_unorm(vol_pred), 20, torch.arange(3000, 8000, 400), True)
                plt.plot(k1, k2, label="Median line", linewidth=4, c="black")
                print(f"Median mean small {np.abs(k1 - k2)[k1 < 5500].mean()}")
                print(f"Median mean large {np.abs(k1 - k2)[k1 >= 5500].mean()}")

                plt.plot((2000, 8500), (2000, 8500), c="green", label="Perfect prediction")
                plt.plot((2000, 8500), (3000, 9500), c="red", label="Error of 1000 [$mm^3$]")
                plt.plot((2000, 8500), (1000, 7500), c="red")
                for season in np.unique(seasons):
                    plt.scatter(datasets.vol_unorm(vol_real[seasons == season]),
                                datasets.vol_unorm(vol_pred[seasons == season]),
                                color=season_to_colors[season], label=season_to_label[season])
                plt.xlabel(r"True volume [$mm^3$]")
                plt.ylabel(r"Predicted volume [$mm^3$]")
                plt.legend()
                plt.tight_layout()
                plt.show()

            return l, season_stats

        if show_plot:
            plt.plot((2000, 8500), (2000, 8500), c="green", label="Perfect prediction")
            plt.plot((2000, 8500), (3000, 9500), c="red", label="Error of 1000 [$mm^3$]")
            plt.plot((2000, 8500), (1000, 7500), c="red")
            plt.scatter(datasets.vol_unorm(vol_real), datasets.vol_unorm(vol_pred))
            plt.xlabel(r"True volume [$mm^3$]")
            plt.ylabel(r"Predicted volume [$mm^3$]")
            plt.legend()
            plt.show()


        
        return l
    
def warmup_multiplicative_schedule(e):
     return 0.01 if e < 40 else min(1/ (e * 0.03), 0.1)
    

def is_better_model(stats, best_stats):
    corr_improve = stats["Correlation"] - best_stats["Correlation"]
    mae_improve =  best_stats["MAE"] - stats["MAE"]
    steep_improve = abs(1 - best_stats["Steepness"]) - abs(1 - stats["Steepness"])
    rate = corr_improve * (50 / 0.03) + mae_improve + steep_improve * (50 / 0.1)

    return rate > 0


def extract_features(dataloader, model):
    model.eval()
    all_features = []
    all_volumes = []

    with torch.no_grad():
        for images, plant, imagemask, label, weight in dataloader:
            images = images.to(device)
            imagemask = imagemask.to(device)

            feats = model(images, imagemask)  # now returns (B, 384)

            all_features.append(feats.cpu())
            all_volumes.append(label.cpu())

    all_features = torch.cat(all_features, dim=0)
    all_volumes = torch.cat(all_volumes, dim=0)

    return all_features, all_volumes

if __name__ == "__main__":
    accelerate.utils.set_seed(0)

    #model_name = "regulatedtransformer-direct.pth"
    #model_name = "self-distill-regulated_transformer_new_mae_rate_original.pth"
    model_name = "kd_regulated_transformer_new2.pth"
    #model_name = "nodistance-regulated_transformer.pth"
    #model_name = "nodistancenoseg-regulated_transformer.pth"
    #model_name = "noaug-regulated_transformer.pth"
    #model_name = "artificial_bestpose_regulated_transformer.pth"
    #model_name = "artificial_bestpose6_regulated_transformer.pth"
    #model_name = "artificial_randompose_regulated_transformer.pth"
    #model_name = "artificial_fipnormal_regulated_transformer.pth"
    #model_name = "artificial_fipuniform_regulated_transformer.pth"
    use_auto_pair = False

    if model_name == "regulatedtransformer-direct.pth" or model_name == "kd_regulated_transformer_new2.pth" or model_name == "self-distill-regulated_transformer_new_mae_rate_original.pth" or model_name == "kd_regulated_transformer_new2.pth":
        if use_auto_pair:
            train_dataset, val_dataset, test_dataset = utils_experiment.get_auto_split_dataset_second()
        else:
            train_dataset, val_dataset, test_dataset = utils_experiment.get_default_image_dataset()
            #train_dataset, val_dataset, test_dataset = utils_experiment.get_raw_image_dataset()
    elif model_name == "nodistance-regulated_transformer.pth":
        train_dataset, val_dataset, test_dataset = utils_experiment.get_default_image_dataset_wo_distance()
        
    elif model_name == "nodistancenoseg-regulated_transformer.pth":
        train_dataset, val_dataset, test_dataset = utils_experiment.get_default_image_dataset_wo_distance_and_seg()
    elif model_name == "noaug-regulated_transformer.pth":
        train_dataset, val_dataset, test_dataset = utils_experiment.get_default_image_dataset_wo_distance_and_seg(disable_augmentation = True)
        train_dataset.random_sequence_len = False
    elif model_name == "artificial_bestpose_regulated_transformer.pth":
        train_dataset, val_dataset, test_dataset = utils_experiment.get_artifical_image_dataset_sideviews()
    elif model_name == "artificial_bestpose6_regulated_transformer.pth":
        train_dataset, val_dataset, test_dataset = utils_experiment.get_artifical_image_dataset_sideviews6()
    elif model_name == "artificial_randompose_regulated_transformer.pth":
        train_dataset, val_dataset, test_dataset = utils_experiment.get_artificial_image_dataset_randompose()
    elif model_name == "artificial_fipnormal_regulated_transformer.pth":
        train_dataset, val_dataset, test_dataset = utils_experiment.get_artificial_dataset_fiplike_normal("image")
    elif model_name == "artificial_fipuniform_regulated_transformer.pth":
        train_dataset, val_dataset, test_dataset = utils_experiment.get_artificial_dataset_fiplike_uniform("image")
    else:
         assert False


    #train_loader = DataLoader(train_dataset, batch_size=32, num_workers=8, shuffle=True, pin_memory=True)
    #val_loader = DataLoader(val_dataset, batch_size=32, num_workers=8, shuffle=False, pin_memory=True, collate_fn=datasets.img_validation_collate_fn)
    #test_loader = DataLoader(test_dataset, batch_size=32, num_workers=8, shuffle=False, pin_memory=True, collate_fn=datasets.img_validation_collate_fn)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,)
    val_loader = DataLoader(val_dataset, batch_size=256,  shuffle=False,  collate_fn=datasets.img_validation_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=256,  shuffle=False, collate_fn=datasets.img_validation_collate_fn)

    if False: # Evaluation using subsets of images
        model = torch.load(helpers.get_exp_path() / f"{model_name}", weights_only=False).to(device).eval()
        ls = []
        ind = []
        for i, (_, row) in enumerate(test_dataset.plant_mapping.iterrows()):
            if len(row["images"]) >= 10:
                ind.append(i)
        print(f"Sel: {len(ind)}")
        for nimg in range(1, 11):
            test_dataset.random_sequence_len = False
            test_dataset.min_seq_len = nimg
            test_dataset.max_seq_len = nimg
            td = Subset(test_dataset, ind)
            test_loader = DataLoader(td, batch_size=256, shuffle=False, collate_fn=datasets.img_validation_collate_fn)
            l = evaluate(test_loader, model, False, fraction_include=1)
            ls.append(l)
        plt.plot(list(range(1, 11)), [l["Correlation"] for l in ls], label="Correlation", linewidth=3)
        plt.plot(list(range(1, 11)), [l["Steepness"] for l in ls], label="Steepness", linewidth=3)
        plt.xlabel("# Images", fontsize=24)
        plt.xticks(list(range(1, 11)), fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(prop={"size": 20})
        plt.tight_layout()
        plt.show()
        exit(0)

    # Plot performance for different levels of removing outliers
    if False:
        model = torch.load(helpers.get_exp_path() / f"{model_name}", weights_only=False).to(device).eval()
        checks = np.linspace(1, 0.3, 30)
        ls = []
        sstats = []
        for v in checks:
            l, sstat = evaluate(test_loader, model, fraction_include=v, detail_eval=True)
            ls.append(l)
            sstats.append(sstat)
        if False:
            for stage in [0, 1, 2]:
                correlation_vals = [entry[stage]['Correlation'] for entry in sstats]
                steepness_vals  = [entry[stage]['Steepness'] for entry in sstats]

                plt.plot(checks, correlation_vals, label=f"stage {stage} - correlation")
                plt.plot(checks, steepness_vals,  label=f"stage {stage} - steepness")
        plt.plot(checks, [l["Correlation"] for l in ls], label="Correlation", linewidth=3)
        plt.plot(checks, [l["Steepness"] for l in ls], label="Steepness", linewidth=3)
        plt.xlabel("Fraction of included predictions", fontsize=24)
        plt.legend(prop={"size": 20})
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim(bottom=0.75)
        plt.tight_layout()
        plt.show()
        exit(0)

    evaluation = True
    if evaluation == True:

        if fine_tuning == False:
            model = torch.load(helpers.get_exp_path() / f"{model_name}", weights_only=False).to(device).eval()
        
        elif fine_tuning == True:
            base_model = models.RegulatedTransformer()
            model = models.RegulatedTransformerWithDINO(
                base_model,
                dino_type="dinov3",
                finetune_last_dino_block=True
            ).to(device)

            #model = models.RegulatedTransformerWithResNet50(
            #    base_model,
            #    finetune_last_resnet_block=True
            #).to(device).eval()

            state = torch.load(helpers.get_exp_path() / model_name, map_location=device)
            model.load_state_dict(state)


        model.eval()


        print("\n=== PURE FORWARD TIMING ===")

        # take one batch
        images, plant, imagemask, label, weight = next(iter(test_loader))
        images = images.to(device)
        imagemask = imagemask.to(device)

        # Warmup (important for CUDA)
        for _ in range(10):
            _ = model(images, imagemask)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Timed runs
        runs = 100
        start = time.time()

        for _ in range(runs):
            _ = model(images, imagemask)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.time()

        time_per_spike = (end - start) / (runs * images.size(0))

        print(f"Pure forward time per spike: {time_per_spike:.6f}s")
        print(f"Pure forward time per spike: {time_per_spike*1000:.3f} ms")


        print_features = True
        if print_features: 

            # ---- switch regulated transformer to feature mode ----
            if hasattr(model, "regulated_transformer"):
                model.regulated_transformer.output = "features"
            else:
                model.output = "features"

            #evaluate(test_loader, model, show_plot=True, fraction_include=1, detail_eval=True, save_test_predictions=helpers.get_exp_path() / f"test_predictions_{model_name}.csv")
            # ---- extract latent features ----
            features, volumes = extract_features(test_loader, model)

            print("Feature shape:", features.shape)
            print("Volume shape:", volumes.shape)

            # Convert to numpy
            features_np = features.numpy()
            volumes_np = volumes.numpy()

            # PCA to 2D
            pca = PCA(n_components=2)
            lat2d = pca.fit_transform(features_np)

            # ---- explained variance ----
            expl_var = pca.explained_variance_ratio_ * 100
            print(f"Explained variance PC1: {expl_var[0]:.2f}%")
            print(f"Explained variance PC2: {expl_var[1]:.2f}%")
            print(f"Cumulative (2 PCs): {(expl_var[0] + expl_var[1]):.2f}%")

            # Convert volume to mm³ for coloring
            vol_mm3 = datasets.vol_unorm(torch.tensor(volumes_np)).numpy()

            plt.figure(figsize=(6,5))
            sc = plt.scatter(
                lat2d[:, 0],
                lat2d[:, 1],
                c=vol_mm3,
                cmap="viridis",
                s=30
            )
            cbar = plt.colorbar(sc)
            cbar.set_label("True volume [mm³]", fontsize=20)
            cbar.ax.tick_params(labelsize=14)
            #plt.xlabel("PC1")
            #plt.ylabel("PC2")
            plt.xlabel(f"PC1 ({expl_var[0]:.1f}%)", fontsize=22)
            plt.ylabel(f"PC2 ({expl_var[1]:.1f}%)", fontsize=22)

            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            plt.xlim(-22, 22)
            plt.ylim(-22, 22)
            #plt.title("Regulated Transformer Latent Space")
            plt.tight_layout()

            save_path = helpers.get_exp_path() / f"pca_{model_name}.png"
            plt.savefig(save_path, dpi=600)
            print(f"Saved PCA plot to {save_path}")

            # Quantify alignment
            corr_pc1 = np.corrcoef(lat2d[:,0], volumes_np)[0,1]
            print("Correlation PC1 vs normalized volume:", corr_pc1)

            # ---- t-SNE ----
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_np)

            tsne = TSNE(
                n_components=2,
                perplexity=30,
                learning_rate=200,
                max_iter=2000,
                init="pca",
                random_state=0
            )

            tsne_2d = tsne.fit_transform(features_scaled)

            plt.figure(figsize=(6,5))
            sc = plt.scatter(
                tsne_2d[:, 0],
                tsne_2d[:, 1],
                c=vol_mm3,
                cmap="viridis",
                s=30
            )
            plt.colorbar(sc, label="True volume [mm³]")
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.tight_layout()

            save_path_tsne = helpers.get_exp_path() / f"tsne_{model_name}.png"
            plt.savefig(save_path_tsne, dpi=300)
            print(f"Saved t-SNE plot to {save_path_tsne}")

            # ---- UMAP ----
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_np)

            umap_model = umap.UMAP(
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                metric="euclidean",
                random_state=0
            )

            umap_2d = umap_model.fit_transform(features_scaled)

            plt.figure(figsize=(6,5))
            sc = plt.scatter(
                umap_2d[:, 0],
                umap_2d[:, 1],
                c=vol_mm3,
                cmap="viridis",
                s=30
            )
            plt.colorbar(sc, label="True volume [mm³]")
            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")
            plt.tight_layout()

            save_path_umap = helpers.get_exp_path() / f"umap_{model_name}.png"
            plt.savefig(save_path_umap, dpi=300)
            print(f"Saved UMAP plot to {save_path_umap}")

        else: 
            evaluate(test_loader, model, show_plot=True, fraction_include=1, detail_eval=True, save_test_predictions=helpers.get_exp_path() / f"test_predictions_{model_name}.csv")
            
    
        exit(0)

    if fine_tuning == False:
        model = models.RegulatedTransformer().to(device)

    elif fine_tuning == True:
        base_model = models.RegulatedTransformer()
        model = models.RegulatedTransformerWithDINO(
            base_model,
            dino_type="dinov3",
            finetune_last_dino_block=True
        ).to(device)

        #model = models.RegulatedTransformerWithResNet50(
        #base_model,
        #finetune_last_resnet_block=True
        #).to(device)



    model.train()
    lossfn = models.RegulatedTransformerLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #optimizer = torch.optim.AdamW(
    #[
    #    {"params": model.regulated_transformer.parameters(), "lr": 1e-3},
    #    {"params": model.dino.blocks[-1].parameters(), "lr": 5e-5},
    #    {"params": model.dino.norm.parameters(), "lr": 5e-5},
    #],
    #weight_decay=1e-4,
    #)
    #optimizer = torch.optim.AdamW(
    #    [
    #        # downstream model
    #        {"params": model.regulated_transformer.parameters(), "lr": 1e-3},
    #        # ResNet layer4 (index 7 in backbone)
    #        {"params": model.backbone[7].parameters(), "lr": 5e-5},
    #        # projection head
    #        {"params": model.proj.parameters(), "lr": 5e-5},
    #    ],
    #    weight_decay=1e-4,
    #)

    #optimizer = torch.optim.AdamW(
    #    [
    #        # downstream model
    #        {"params": model.regulated_transformer.parameters(), "lr": 1e-3},
    #
    #        # DINOv3 last block
    #        {"params": model.dino.blocks[-1].parameters(), "lr": 5e-5},
    #
    #        # DINOv3 norm
    #        {"params": model.dino.norm.parameters(), "lr": 5e-5},
    #    ],
    #    weight_decay=1e-4,
    #)

    scheduler = scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_multiplicative_schedule)
    import time
    total_train_time = 0
    n_train_spikes = len(train_dataset)
    best_val = None
    best_model = None
    epoch_times = []
    for epoch in range(500):
        errs_train = []
        model.train()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_start = time.time()
        for images, label, imagemask, weight in train_loader:
            images, label, imagemask, weight = [t.to(device) for t in [images, label, imagemask, weight]]

            x = model(images, imagemask)
                
            loss = lossfn(x, label, weight)
            errs_train.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_end = time.time()

        epoch_time = epoch_end - epoch_start
        epoch_times.append(epoch_time)
        print(f"Epoch {epoch} training time: {epoch_time:.2f}s")
        print(f"Time per spike: {epoch_time / len(train_dataset):.6f}s")

        err_val = evaluate(val_loader, model)
        if best_val is None or (is_better_model(err_val, best_val) and epoch > 400):
            best_val = err_val
            #best_model = copy.deepcopy(model.state_dict())
            best_model = copy.deepcopy(model).cpu()
            best_epoch = epoch

        print(f"epoch {epoch}. Train: {np.mean(errs_train)}")

    print("\n=== Training Time Statistics ===")
    print(f"Mean time per epoch: {np.mean(epoch_times):.2f}s")
    print(f"Std time per epoch: {np.std(epoch_times):.2f}s")
    print(f"Min epoch time: {np.min(epoch_times):.2f}s")
    print(f"Max epoch time: {np.max(epoch_times):.2f}s")

    print(f"\n Best stats {best_val}")
    torch.save(best_model, helpers.get_exp_path() / f"{model_name}")


    #{'MAE': 675.1566772460938, 'Correlation': 0.7558175552055129, 'Steepness': 0.7455614340258181, 'MAPE': 15.356525778770447, 'Loss': 0.19118213653564453}
    #{'MAE': 712.8218994140625, 'Correlation': 0.7463525798172536, 'Steepness': 0.7593150355364198, 'MAPE': 15.976999700069427, 'Loss': 0.20875975489616394}
    
    #4096:8
    #{'MAE': 675.1566772460938, 'Correlation': 0.7558175552055129, 'Steepness': 0.7455614340258181, 'MAPE': 15.356525778770447, 'Loss': 0.19118213653564453}
    
    #16:8
    #{'MAE': 666.13671875, 'Correlation': 0.7570779852580791, 'Steepness': 0.7263923261101991, 'MAPE': 15.119819343090057, 'Loss': 0.18902479112148285}

    #creating .venv
    #16:8
    #{'MAE': 723.8834228515625, 'Correlation': 0.7552360497584256, 'Steepness': 0.8298041411096486, 'MAPE': 16.433194279670715, 'Loss': 0.19541175663471222}
    #4096:8
    #{'MAE': 669.3644409179688, 'Correlation': 0.7539337938001187, 'Steepness': 0.7362493785673844, 'MAPE': 15.144048631191254, 'Loss': 0.1944139450788498}
    #16:8
    #{'MAE': 723.8834228515625, 'Correlation': 0.7552360497584256, 'Steepness': 0.8298041411096486, 'MAPE': 16.433194279670715, 'Loss': 0.19541175663471222}
    #{'MAE': 669.3644409179688, 'Correlation': 0.7539337938001187, 'Steepness': 0.7362493785673844, 'MAPE': 15.144048631191254, 'Loss': 0.1944139450788498}


#pip freeze > requirements.lock.txt
#source .venv/bin/activate
#export CUBLAS_WORKSPACE_CONFIG=:16:8
#final
#{'MAE': 666.13671875, 'Correlation': 0.7570779852580791, 'Steepness': 0.7263923261101991, 'MAPE': 15.119819343090057, 'Loss': 0.18902479112148285}


#source .venv/bin/activate
#pip install -e .

#models are saved in volume prediction fip folder not inside .venv
#{'MAE': 680.212646484375, 'Correlation': 0.7502734676847155, 'Steepness': 0.7305241176451045, 'MAPE': 15.36526083946228, 'Loss': 0.19476602971553802} 

#alles mit 4096 erstellt:
#{'MAE': 676.26806640625, 'Correlation': 0.7527568879821801, 'Steepness': 0.7255946111598219, 'MAPE': 15.276043117046356, 'Loss': 0.19275256991386414}

#{'MAE': 675.501708984375, 'Correlation': 0.7520244931837409, 'Steepness': 0.7484986829231086, 'MAPE': 15.322299301624298, 'Loss': 0.1904110461473465}


#open new tmux, type 16:8, wihtout activate .venv
#{'MAE': 654.3128051757812, 'Correlation': 0.7561503121698537, 'Steepness': 0.712138686044237, 'MAPE': 14.797529578208923, 'Loss': 0.19091705977916718}

#new yolo file: 
#Best stats {'MAE': 575.9115600585938, 'Correlation': 0.7711124579205426, 'Steepness': 0.8295818984075184, 'MAPE': 12.662704288959503, 'Loss': 0.5147190690040588}
#{'MAE': 694.15576171875, 'Correlation': 0.7692312056069635, 'Steepness': 0.8129156604149921, 'MAPE': 15.546469390392303, 'Loss': 0.18737353384494781}


#{'MAE': 654.3128051757812, 'Correlation': 0.7561503121698537, 'Steepness': 0.712138686044237, 'MAPE': 14.797529578208923, 'Loss': 0.19091705977916718}

#Fomo: 
#{'MAE': 1028.49609375, 'Correlation': 0.5184460775649676, 'Steepness': 0.5865879402429743, 'MAPE': 23.533296585083008, 'Loss': 0.42921316623687744}

#dinov3: 
#{'MAE': 665.1604614257812, 'Correlation': 0.7717988657840211, 'Steepness': 0.7565330127454333, 'MAPE': 14.62927758693695, 'Loss': 0.1825399249792099}

#self distillation with dinov2: 
#{'MAE': 643.9614868164062, 'Correlation': 0.8262098217850918, 'Steepness': 0.9434035922537167, 'MAPE': 14.732643961906433, 
# 'Loss': 0.1428295224905014}#
