import torch
import copy
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from volume_prediction_fip.experiments_volume_models.shared import (
    utils_experiment, models, utils_3d, datasets
)
from volume_prediction_fip.utils import helpers

import accelerate
accelerate.utils.set_seed(0)

def warmup_multiplicative_schedule(e):
     return 0.01 if e < 40 else min(1/ (e * 0.03), 0.1)



#def warmup_multiplicative_schedule(e):
#    if e < 10:
#        return (e + 1) / 10        # linear warmup 0.1 → 1.0
#    return 1.0   

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ----------------------------
# LOSS FUNCTIONS
# ----------------------------
#loss_KD: mse_loss / mse_l2_loss / cosine_loss / None to disable feature KD

use_label_kd = False      
loss_KD = "mse_l2_loss"    
label_kd_weight = 0

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
    helpers.get_exp_path() / "3dglobimgensemble_new.pth",
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
optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
scheduler = scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_multiplicative_schedule)


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

        for images, points, imagemask, pointmask, label, weight in dataloader:

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
        
        loss = ((vol_pred-vol_real) ** 2 * weights).mean().item()

        return {
            "MAE": mae,
            "Correlation": corr,
            "Steepness": slope[0], 
            "MAPE": mape, 
            "Loss": loss
        }

# ----------------------------
# TEST
# ----------------------------
evaluation = True
if evaluation == True:
    best_model = torch.load(helpers.get_exp_path() / "kd_regulated_transformer_new2.pth", weights_only=False).to(device).eval()
    best_model.eval()
    
    test_stats = evaluate(test_loader, best_model.to(device))
    print("test:", test_stats)
    exit(0)

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
            points[:, :, 0:3], points[:, :, 6], num_samples=None,
        bins=10)

        # ---- teacher features ----
        #teacher model returns image + point cloud embeddings, shape: (batch, 256)
        with torch.no_grad():
            teacher_feat = teacher.img_net(images, imagemask)
            teacher_feat = teacher.prec_img(teacher_feat)
            point_feat = teacher.point_net(points_ri, pointmask)
            teacher_feat = torch.cat((teacher_feat, point_feat), dim=-1)


        # ---- student supervised loss ----
        #loss for volume prediction
        student_out = student(images, imagemask)
        loss_reg = lossfn(student_out, label, weight)


        if use_label_kd:
            with torch.no_grad():
                teacher_out = teacher(images, imagemask, points_ri, pointmask)
            student_pred = student_out[2]   # tpred
            loss_label_kd = F.mse_loss(student_pred, teacher_out)
        else:
            loss_label_kd = 0.0

        # ---- student KD features ----
        #loss to match teacher features
        #student_feat = student(images, imagemask, output="features") #(batch, 384)
        # ADD this (toggle output mode explicitly)
        # ---- student KD features (STABLE, no dropout) ----
        student.eval()
        
        student.output = "features"
        student_feat = student(images, imagemask)
        student.train()
        student.output = "volume"


        student_feat_proj = student.kd_proj(student_feat) #384 -> 256

        if loss_KD is None:
            loss_kd = 0.0
        elif loss_KD == "mse_loss":
            loss_kd = F.mse_loss(student_feat_proj, teacher_feat)
        elif loss_KD == "mse_l2_loss": 
                # ---- direction (geometry) ----
            student_dir = F.normalize(student_feat_proj, dim=1)
            teacher_dir = F.normalize(teacher_feat, dim=1)

            dir_loss = ((student_dir - teacher_dir) ** 2).sum(dim=1).mean()

            # ---- magnitude (scale / volume cue) ----
            norm_loss = F.mse_loss(
                student_feat_proj.norm(dim=1),
                teacher_feat.norm(dim=1)
            )

            # ---- combined KD loss ----
            loss_kd = dir_loss + 0.2 * norm_loss

        elif loss_KD == "cosine_loss":
            # ---- feature KD: direction-only cosine loss ----
            student_dir = F.normalize(student_feat_proj, dim=1)
            teacher_dir = F.normalize(teacher_feat.detach(), dim=1)

            loss_kd = 1.0 - F.cosine_similarity(student_dir, teacher_dir, dim=1).mean()



        # ---- KD warmup ----
        kd_weight = min(1.0, max(0.0, (epoch - 20) / 20))

        # ---- total loss ----
        loss = loss_reg + 0.2 * kd_weight * loss_kd + label_kd_weight * loss_label_kd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        errs.append(loss.item())
    scheduler.step()

    print(f"epoch {epoch:03d} | train loss {np.mean(errs):.4f}")

    # ----------------------------
    # VALIDATION (NO KD)
    # ----------------------------
    val_stats = evaluate(val_loader, student)
    print("val:", val_stats)

    if best_val is None or (val_stats["Loss"] < best_val["Loss"] and epoch > 400):

        best_val = val_stats
        best_model = copy.deepcopy(student).cpu()



# ----------------------------
# SAVE BEST MODEL
# ----------------------------
print("Saving model with output mode:", student.output)

torch.save(
    best_model,
    helpers.get_exp_path() / "kd_regulated_transformer_new2.pth"
)

print("\nBEST VALIDATION:")
print(best_val)






#mse loss: 
#BEST VALIDATION:                                                               
#{'MAE': 572.1832275390625, 'Correlation': 0.7952069548499624, 'Steepness': 
# 0.8012752913329855, 'MAPE': 12.327175587415695, 'Loss': 0.48768362402915955}  
#      
#test: {'MAE': 637.4743041992188, 'Correlation': 0.7675991672824534, 'Steepness'
#: 0.6714089973915504, 'MAPE': 13.888303935527802, 'Loss': 0.19660817086696625}

#mse_l2-loss: 
#BEST VALIDATION:
#{'MAE': 607.0867919921875, 'Correlation': 0.7754453838235537, 'Steepness': 0.715006413870725, 'MAPE': 12.992222607135773, 'Loss': 0.49433284997940063}
#test: {'MAE': 635.0047607421875, 'Correlation': 0.7531962499711902, 'Steepness': 0.6153654971837877, 'MAPE': 13.807308673858643, 'Loss': 0.23760971426963806}


#BEST VALIDATION:
#{'MAE': 589.093505859375, 'Correlation': 0.8104776974649247, 'Steepness': 0.9055227816675178, 'MAPE': 12.085027247667313, 'Loss': 0.506040096282959}
#test: {'MAE': 649.3827514648438, 'Correlation': 0.7698646326660973, 'Steepness': 0.7424262737804725, 'MAPE': 14.222273230552673, 'Loss': 0.19141821563243866}

#BEST VALIDATION:
#{'MAE': 577.8952026367188, 'Correlation': 0.7881406404491325, 'Steepness': 0.7681445624772641, 'MAPE': 12.118349224328995, 'Loss': 0.4783382713794708}
#test: {'MAE': 633.3038940429688, 'Correlation': 0.7520576612178477, 'Steepness': 0.6213431421746668, 'MAPE': 13.870304822921753, 'Loss': 0.21718622744083405}
#


#normal mse loss: 
#BEST VALIDATION:
#{'MAE': 602.1560668945312, 'Correlation': 0.7757290974833144, 'Steepness': 0.7628026050056109, 'MAPE': 12.907113134860992, 'Loss': 0.5203903913497925}
#test: {'MAE': 646.9292602539062, 'Correlation': 0.7626439963832992, 'Steepness': 0.7032790680481817, 'MAPE': 14.26791101694107, 'Loss': 0.2018510401248932}

#mse_l2_loss:
#{'MAE': 614.831298828125, 'Correlation': 0.7886355418112179, 'Steteepness': 0.8540056128084437, 'MAPE': 13.16368281841278, 'Loss 0': 0.5506218671798706}
#test: {'MAE': 660.1543579101562, 'Correlation': 0.771857217221851513, 'Steepness': 0.7696720382621768, 'MAPE': 14.68489170074462 '9, 'Loss': 0.19422726333141327}

#cosine loss: 
#{'MAE': 618.8182373046875, 'Correlation': 0.7681069554445752, 'Steepness': 0.7956278904452685, 'MAPE': 12.938842177391052, 'Loss': 0.5609971284866333}
#test: {'MAE': 618.1915283203125, 'Correlation': 0.7932337657793812, 'Steepness': 0.6725620344456832, 'MAPE': 13.559719920158386, 'Loss': 0.18399187922477722}




# now KD start later, not at epoch 0:
# mse loss: 
#{'MAE': 589.1156616210938, 'Correlation': 0.788512945498271, 'Steepness': 0.7655279876311607, 'MAPE': 12.17295303940773, 'Loss': 0.4786664843559265}
#test: {'MAE': 620.5490112304688, 'Correlation': 0.781344842106077, 'Steepness': 0.6554179625793761, 'MAPE': 13.604074716567993, 'Loss': 0.19899366796016693}

#mse_l2
#BEST VALIDATION:
#{'MAE': 575.5912475585938, 'Correlation': 0.8099520827712295, 'Steepness': 0.8249110979964706, 'MAPE': 12.1636763215065, 'Loss': 0.45435255765914917}
#test: {'MAE': 604.5501098632812, 'Correlation': 0.7934696312180028, 'Steepness': 0.725873866778188, 'MAPE': 13.335569202899933, 'Loss': 0.16710416972637177}


#cosine loss: 
#BEST VALIDATION:
#{'MAE': 595.3980712890625, 'Correlation': 0.7703121068549821, 'Steepness': 0.7728897076330767, 'MAPE': 12.540744245052338, 'Loss': 0.5350524187088013}
#test: {'MAE': 614.9932250976562, 'Correlation': 0.7937365067489892, 'Steepness': 0.6615145758978718, 'MAPE': 13.310515880584717, 'Loss': 0.18551741540431976}#


#Above: ground truth loss + feature loss (0.2)
#Below: ground truth loss + soft labels loss (0.5)
#lets try different weights for the soft label loss: 

###############################################################################

#with warmup

#mse_l2 loss --> taking this loss everywhere
#after fixing the mode: 
#BEST VALIDATION:
#{'MAE': 627.2119750976562, 'Correlation': 0.7916009446748723, 'Steepness': 0.8617235891701068, 'MAPE': 13.33275735378, 'Loss': 0.546151876449585
#test: {'MAE': 665.8397827148438, 'Correlation': 0.7779677590593753, 'Steepness': 0.7783737054915475, 'MAPE': 14.60893303155899, 'Loss': 0.18911725282669067}

#mse loss: 
#{'MAE': 621.4829711914062, 'Correlation': 0.7672124049236879, 'Steepness': 0.7344180632746226, 'MAPE': 13.40995579957962, 'Loss': 0.5265281796455383}
#test: {'MAE': 610.1074829101562, 'Correlation': 0.7718404588465023, 'Steepness': 0.6429970473066174, 'MAPE': 13.575607538223267, 'Loss': 0.2036481499671936}

#cosine: 
#BEST VALIDATION:
#{'MAE': 630.9806518554688, 'Correlation': 0.777174909846401, 'Steepness': 0.8211285638716439, 'MAPE': 13.630974292755127, 'Loss': 0.5586931705474854}
#test: {'MAE': 659.3699951171875, 'Correlation': 0.7628622327338725, 'Steepness': 0.7379834530443211, 'MAPE': 14.809118211269379, 'Loss': 0.1933498978614807}

#same LR: 
#test: {'MAE': 655.72314453125, 'Correlation': 0.7681468312253636, 'Steepness': 0.7704662473145397, 'MAPE': 14.626525342464447, 'Loss': 0.18492993712425232}



##############################################################
#new ensemble model:
#BEST VALIDATION:
#{'MAE': 626.5914916992188, 'Correlation': 0.7717832280215127, 'Steepness': 0.796425784679147, 'MAPE': 13.65472823381424, 'Loss': 0.571654200553894}
#{'MAE': 639.9292602539062, 'Correlation': 0.7700463979965172, 'Steepness': 0.7193728351203522, 'MAPE': 14.595045149326324, 'Loss': 0.17490780353546143}

#{'MAE': 633.6015014648438, 'Correlation': 0.7757327546121104, 'Steepness': 0.8612080553836232, 'MAPE': 13.461072742938995, 'Loss': 0.600167989730835}
#test: {'MAE': 685.2740478515625, 'Correlation': 0.7636669126718427, 'Steepness': 0.792729142932506, 'MAPE': 15.349675714969635, 'Loss': 0.1845831423997879}