import torch
from torch.utils.data import DataLoader
from volume_prediction_fip.experiments_volume_models.shared import utils_experiment, models, utils_3d
from volume_prediction_fip.utils import helpers

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ---- dataset (must be combined) ----
train_dataset, val_dataset, test_dataset = utils_experiment.get_combined_image_point_dataset()
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# ---- teacher (frozen) ----
teacher = torch.load(
    helpers.get_exp_path() / "3dglobimgensemble.pth",
    weights_only=False
).to(device).eval()

# ---- student ----
student = models.RegulatedTransformer().to(device)
