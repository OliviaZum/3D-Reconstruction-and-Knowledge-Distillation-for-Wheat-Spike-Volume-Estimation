"""
Use an ensemble of linear models on single images to predict volume. (Use median to predict final volume)

No efforts have been made to make this nice or fast since it anyway works quite badly.
{'MAE': 674.6136474609375, 'Correlation': 0.7208296074639524, 'Steepness': 0.41145407064369877}
"""

from sklearn.linear_model import LinearRegression
import numpy as np
from typing import List
import torch
from experiments_volume_models.shared import datasets, utils_experiment
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F

class LinearPredictorSingleImage:
    """
    Creates a linear ensemble on single image tokens. Results on images are combined via mean.
    """
    def __init__(self) -> None:
        super().__init__()
        self.linregs: List[LinearRegression] = []

    def train_call(self, x: torch.Tensor, mask: torch.Tensor, y: torch.Tensor):
        weighted = False
        
        pred = LinearRegression()
        img_features = x[~mask]
        mlen = (~mask).sum(dim=1)
        y_labels = y.repeat_interleave(mlen)
        if weighted:
            weights = np.array([ 7.        ,  1.36744186,  1.        ,  1.1484375 ,  2.64864865,
                6.        , 49.        ])
            bins = datasets.vol_norm(np.array(list(range(2000, 9001, 1000))))
            sample_weight = np.zeros(y_labels.shape)
            for i in range(len(bins) - 1):
                wu= y_labels > bins[i]
                wb = y_labels < bins[i+1]
                mask = wu & wb
                sample_weight[mask] = weights[i]
        else:
            sample_weight = None
        pred.fit(img_features.cpu().numpy(), y_labels.cpu().numpy(), sample_weight=sample_weight)
        self.linregs.append(pred)

    def __call__(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            img_features = x[~mask]
            mlen = (~mask).sum(dim=1)
            ys = []
            for pred in self.linregs:
                u = torch.tensor(pred.predict(img_features.cpu().numpy()))
                offset = 0
                result = torch.zeros(mlen.shape[0])
                for i in range(mlen.shape[0]):
                    result[i] = torch.mean(u[offset:(offset + mlen[i])])
                    offset += mlen[i]
                ys.append(result)
            
            y = torch.stack(ys, dim=1).to(x.device)
            y = torch.mean(y, dim=1, keepdim=True)
            return y

def evaluate(model, dataloader: DataLoader, show_plot = False, should_print=True, ignore_outliers = False):
    with torch.no_grad():
        vol_pred = []
        vol_real = []
        for images, plant, imagemask, label, _ in dataloader:
            
            volume = model(images, imagemask).squeeze()
            vol_pred.append(volume)
            vol_real.append(label)
        vol_pred = torch.concat(vol_pred)
        vol_real = torch.concat(vol_real)

        l = {}
        l["MAE"] = F.l1_loss(datasets.vol_unorm(vol_pred), datasets.vol_unorm(vol_real)).item()
        l["Correlation"] = np.corrcoef(vol_real, vol_pred)[0, 1]
        A = np.vstack([vol_real, np.ones(len(vol_real))]).T
        linregcoeff, _, _, _ = np.linalg.lstsq(A, vol_pred, rcond=None)
        l["Steepness"] = linregcoeff[0]
        if should_print:
             print(l)
        if show_plot:
            plt.plot((2000, 8500), (2000, 8500))
            plt.plot((2000, 8500), (3000, 9500), c="red")
            plt.plot((2000, 8500), (1000, 7500), c="red")
            plt.scatter(datasets.vol_unorm(vol_real), datasets.vol_unorm(vol_pred))
            plt.show()
            
            """
            plt.scatter(torch.abs(vol_real - vol_pred), torch.exp(logvars))
            plt.show()
            """
        
        return l

if __name__ == "__main__":
    model = LinearPredictorSingleImage()

    train_dataset, val_dataset, test_dataset = utils_experiment.get_default_image_dataset()

    train_dataset.random_sequence_len = False
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, collate_fn=datasets.img_validation_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, collate_fn=datasets.img_validation_collate_fn)

    with torch.no_grad():
        for i in range(50):
            features = []
            labels = []
            mask = []
            for features_, labels_, mask_, _ in train_loader:
                features.append(features_)
                labels.append(labels_)
                mask.append(mask_)
            features = torch.concat(features, dim=0)
            labels = torch.concat(labels, dim=0)
            mask = torch.concat(mask, dim=0)

            erase_20 = torch.rand(mask.shape[0]) < 0.8
            features = features[erase_20]
            labels = labels[erase_20]
            mask = mask[erase_20]

            model.train_call(features, mask, labels)

    evaluate(model, test_loader, True)