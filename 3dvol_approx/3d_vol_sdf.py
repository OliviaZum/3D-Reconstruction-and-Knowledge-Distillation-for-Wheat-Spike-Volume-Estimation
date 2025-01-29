import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import trimesh
import numpy as np
import pandas as pd
from pathlib import Path
import pyrender
import copy
import pymeshlab
import tqdm
import pandas as pd
import ast
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
POINT_CLOUD_SIZE = 2000

def load_and_process(file, get_normals=False):
    ply_data = trimesh.load(file)
    #ply_data.apply_translation(-ply_data.center_mass)
    #ply_data.vertices -= ply_data.vertices.mean(axis=0, keepdims=True)
    pcl, face_index = trimesh.sample.sample_surface(ply_data, POINT_CLOUD_SIZE)
    if get_normals:
        normals = ply_data.face_normals[face_index]
        pcl = np.concatenate((pcl, normals), 1)
    return pcl

class PlyDataset(Dataset):
    def __init__(self, models, tensor_path = None, load_tensor = False):
        weird_stuff = set(["2_10_8.ply", "10_9_5.ply", "11_8_9.ply", "6_9_1.ply"])
        df = pd.read_csv(models)
        self.df = df.loc[~df["ply_file"].apply(lambda x: Path(x).name).isin(weird_stuff), :]
        if not load_tensor:
            tensors = []
            for file in self.df["ply_file"].apply(Path).tolist():
                pcl = load_and_process(file, True)
                """
                r = np.array([[ 0.16986588,  0.94465781,  0.28065495],
                    [ 0.54707813,  0.14648281, -0.82416521],
                    [-0.81966524,  0.29353774, -0.4919192 ]])
                pcl[:, :3] = (r @ pcl[:, :3].T).T + [2, 3, -2]
                pcl[:, 3:] = (r @ pcl[:, 3:].T).T
                """
                tensors.append(torch.Tensor(pcl).unsqueeze(0))
            tensors = torch.concat(tensors, dim=0)
            if tensor_path is not None:
                torch.save(tensors, tensor_path)
        else:
            tensors = torch.load(tensor_path, weights_only=True)
        self.data = tensors

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        samples = self.data[idx]
        return idx, samples
    
class SdfDecoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l = nn.Sequential(
            nn.Linear(128 + 3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, idx, positions, embeddings):
        emb = embeddings(idx)
        emb = emb.unsqueeze(1).repeat(1, positions.shape[1], 1)
        v = torch.concat((emb, positions), dim=2)
        x = self.l(v)
        return x

def oriented_point_to_sdf(batch, etas = [0.1, 0.2]):
    p, n = batch[:, :, :3], batch[:, :, 3:]
    pos = []
    vs = []
    for eta in etas:
        o = p + n * eta
        i = p - n * eta
        pos.extend((o, i))
        vs.extend((torch.ones(o.shape, device=device) * eta, -torch.ones(i.shape, device=device) * eta))
    pos = torch.concat(pos, dim=1)
    vs = torch.concat(vs, dim=1)
    return pos, vs


def train_latent_sdf():
    train_dataset = PlyDataset("3dvol_approx/train_ply.csv", "3dvol_approx/train_cache.pth", True)
    val_dataset = PlyDataset("3dvol_approx/val_ply.csv", "3dvol_approx/val_cache.pth", True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = SdfDecoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, 0.5)
    embeddings = nn.Embedding(len(train_dataset), 128, device=device)

    best_val = None
    best_model = None
    for epoch in range(151):
        errs_train = []
        model.train()
        for idx, batch in train_loader:
            batch = batch.to(device)
            idx = idx.to(device)
            pos, vs = oriented_point_to_sdf(batch)

            pred = model(idx, pos, embeddings)
            loss = torch.abs(torch.clamp(pred, -0.2, 0.2) - torch.clamp(vs, -0.2, 0.2)).mean()

            loss.backward()
            errs_train.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()
        #scheduler.step()
        print(f"epoch {epoch}. Train: {np.mean(errs_train)}")
        
        """
        if epoch % 30 == 0 and epoch > 0:
            err_val = evaluate(val_loader, model, "chamfer")

            if best_val is None or err_val < best_val:
                best_val = err_val
                best_model = copy.deepcopy(model).cpu()

            print(f"epoch {epoch}. Val: {err_val}, Train: {np.mean(errs_train)}")
        else:
            print(f"epoch {epoch}. Train: {np.mean(errs_train)}")
        """

    #torch.save(best_model, "3dvol_approx/model_backup_with_pose.pth")

if __name__ == "__main__":
    train_latent_sdf()