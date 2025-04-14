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


def evaluate(dataloader: DataLoader, model: nn.Module, show_plot = False, should_print=True, fraction_include = 1, detail_eval = False):
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
            
            volume, logvar = model(images, imagemask)
            volume = volume.cpu().squeeze()
            vol_pred.append(volume)
            vol_real.append(label)
            weights.append(weight)
            logvars.append(logvar.cpu())
            for p in plant:
                plants.append(p["plant_id"])
        vol_pred = torch.concat(vol_pred)
        vol_real = torch.concat(vol_real)
        weights = torch.concat(weights)
        logvars = torch.concat(logvars)
        
        if fraction_include < 1:
            e = torch.abs(vol_pred - vol_real)
            p95 = torch.quantile(e, fraction_include)
            mask = e < p95
            vol_pred, vol_real, weights, logvars = [a[mask] for a in [vol_pred, vol_real, weights, logvars]]
            plants = [p for m, p in zip(mask, plants) if m]

        l = get_stats(vol_pred, vol_real)
        l["Loss"] = ((vol_pred - vol_real) ** 2 * weights).mean().item()
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

if __name__ == "__main__":
    accelerate.utils.set_seed(0)

    #model_name = "regulatedtransformer-direct.pth"
    model_name = "self-distill-regulated_transformer.pth"
    #model_name = "nodistance-regulated_transformer.pth"
    #model_name = "nodistancenoseg-regulated_transformer.pth"
    #model_name = "noaug-regulated_transformer.pth"
    #model_name = "artificial_bestpose_regulated_transformer.pth"
    #model_name = "artificial_bestpose6_regulated_transformer.pth"
    #model_name = "artificial_randompose_regulated_transformer.pth"
    #model_name = "artificial_fipnormal_regulated_transformer.pth"
    #model_name = "artificial_fipuniform_regulated_transformer.pth"
    use_auto_pair = True

    if model_name == "regulatedtransformer-direct.pth" or model_name == "self-distill-regulated_transformer.pth":
        if use_auto_pair:
            train_dataset, val_dataset, test_dataset = utils_experiment.get_auto_split_dataset_second()
        else:
            train_dataset, val_dataset, test_dataset = utils_experiment.get_default_image_dataset()
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


    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, collate_fn=datasets.img_validation_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, collate_fn=datasets.img_validation_collate_fn)

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

    if True:
        model = torch.load(helpers.get_exp_path() / f"{model_name}", weights_only=False).to(device).eval()
        evaluate(test_loader, model, show_plot=True, fraction_include=1, detail_eval=True)
        exit(0)

    model = models.RegulatedTransformer().to(device)
    lossfn = models.RegulatedTransformerLoss()
    
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
                
            loss = lossfn(x, label, weight)
            errs_train.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        err_val = evaluate(val_loader, model)

        if best_val is None or (is_better_model(err_val, best_val) and epoch > 400):
            best_val = err_val
            best_model = copy.deepcopy(model).cpu()
            best_epoch = epoch

        print(f"epoch {epoch}. Train: {np.mean(errs_train)}")

    print(f"\n Best stats {best_val}")
    torch.save(best_model, helpers.get_exp_path() / f"{model_name}")