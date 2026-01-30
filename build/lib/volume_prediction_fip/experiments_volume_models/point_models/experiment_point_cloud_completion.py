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

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":
    # Results vary quite a bit depending on seed
    accelerate.utils.set_seed(1, deterministic=True)
    
    model_name = "3dcompletion.pth"
    
    if model_name == "3dcompletion.pth":
        train_dataset, val_dataset, test_dataset = utils_experiment.get_ply_dataset(force_recompute=False, voxelize=True)

    print(train_dataset)
   
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
