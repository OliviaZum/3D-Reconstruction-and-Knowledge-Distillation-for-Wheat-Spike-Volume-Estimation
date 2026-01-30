import torch
import copy
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from volume_prediction_fip.experiments_volume_models.shared import (
    utils_experiment, models, utils_3d, datasets
)
from volume_prediction_fip.utils import helpers

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ----------------------------
# DATASETS
# ----------------------------
#load combined dataset: images and point cloud for each spike
#each batch contains images, points, imagemask, pointmask, label, weight
train_dataset, val_dataset, test_dataset = utils_experiment.get_combined_image_point_dataset()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ----------------------------
# TEACHER (FROZEN)
# ----------------------------
#load the pre-trained ensemble
#disable droput, teacher will just run forward, no backprop
teacher = torch.load(
    helpers.get_exp_path() / "3dglobimgensemble.pth",
    weights_only=False
).to(device).eval()

for p in teacher.parameters():
    p.requires_grad_(False)

# ----------------------------
# STUDENT
# ----------------------------
#load student model,regulated transformer, predicts per image predictions / final volume token
student = models.RegulatedTransformer().to(device)

#supervised loss
lossfn = models.RegulatedTransformerLoss()
optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

# ----------------------------
# EVALUATION (reuse style)
# ----------------------------
#no gradients
def evaluate(dataloader, model):
    model.eval()
    with torch.no_grad():
        vol_pred = []
        vol_real = []
        weights  = []

        for images, plant, imagemask, label, weight in dataloader:
            images = images.to(device)
            imagemask = imagemask.to(device)

            pred, _ = model(images, imagemask) #calls student forward, returns tpred, tconf
            pred = pred.cpu().squeeze()

            vol_pred.append(pred)
            vol_real.append(label)
            weights.append(weight)

        vol_pred = torch.cat(vol_pred)
        vol_real = torch.cat(vol_real)
        weights  = torch.cat(weights)

        #compute MAE / correlation / steepness
        mae = F.l1_loss(
            datasets.vol_unorm(vol_pred),
            datasets.vol_unorm(vol_real)
        ).item()

        corr = np.corrcoef(vol_real, vol_pred)[0, 1]

        A = np.vstack([vol_real, np.ones(len(vol_real))]).T
        slope, _, _, _ = np.linalg.lstsq(A, vol_pred, rcond=None)

        vp = datasets.vol_unorm(vol_pred)
        vr = datasets.vol_unorm(vol_real)

        mape = (torch.abs((vr - vp) / (vr + 1e-8))).mean().item() * 100


        return {
            "MAE": mae,
            "Correlation": corr,
            "Steepness": slope[0], 
            "MAPE": mape
        }

# ----------------------------
# TRAINING
# ----------------------------
best_val = None
best_model = None

for epoch in range(500):
    student.train() #enables dropout
    errs = []

    for images, points, imagemask, pointmask, label, weight in train_loader:
        images = images.to(device)
        imagemask = imagemask.to(device)
        points = points.to(device)
        pointmask = pointmask.to(device)
        label = label.to(device)
        weight = weight.to(device)

        # ---- teacher preprocessing ----
        #get bins from point clouds 
        points_ri = utils_3d.to_rigid_invariant_representation(
            points[:, :, 0:3], points[:, :, 6], num_samples=None
        )

        # ---- teacher features ----
        #teacher model returns image + point cloud embeddings, shape: (batch, 256)
        with torch.no_grad():
            teacher_feat = teacher(
                images,
                imagemask,
                points_ri,
                pointmask,
                return_features=True
            )

        # ---- student supervised loss ----
        #loss for volume prediction
        student_out = student(images, imagemask)
        loss_reg = lossfn(student_out, label, weight)

        # ---- student KD features ----
        #loss to match teacher features
        student_feat = student(images, imagemask, output="features") #(batch, 384)
        student_feat_proj = student.kd_proj(student_feat) #384 -> 256

        loss_kd = F.mse_loss(student_feat_proj, teacher_feat)

        # ---- total loss ----
        loss = loss_reg + 0.1 * loss_kd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        errs.append(loss.item())

    print(f"epoch {epoch:03d} | train loss {np.mean(errs):.4f}")

    # ----------------------------
    # VALIDATION (NO KD)
    # ----------------------------
    val_stats = evaluate(val_loader, student)
    print("val:", val_stats)

    if best_val is None or val_stats["MAE"] < best_val["MAE"]:
        best_val = val_stats
        best_model = copy.deepcopy(student).cpu()

# ----------------------------
# SAVE BEST MODEL
# ----------------------------
torch.save(
    best_model,
    helpers.get_exp_path() / "kd_regulated_transformer.pth"
)

# ----------------------------
# TEST
# ----------------------------
test_stats = evaluate(test_loader, best_model.to(device))
print("test:", test_stats)
