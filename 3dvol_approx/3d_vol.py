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

POINT_CLOUD_SIZE = 2000
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_and_process(file, get_normals=False):
    ply_data = trimesh.load(file)
    ply_data.apply_translation(-ply_data.center_mass)
    #ply_data.vertices -= ply_data.vertices.mean(axis=0, keepdims=True)
    pcl, face_index = trimesh.sample.sample_surface(ply_data, POINT_CLOUD_SIZE)
    if get_normals:
        normals = ply_data.face_normals[face_index]
        pcl = np.concatenate((pcl, normals), 1)
    return pcl

def visualize_point_clouds(points_red, points_blue = None):
    scene = pyrender.Scene()

    if points_red is not None:
        red_cloud = trimesh.points.PointCloud(points_red, colors=[255, 0, 0])
        red_mesh = pyrender.Mesh.from_points(points_red, colors=red_cloud.colors)
        scene.add(red_mesh)

    if points_blue is not None:
        blue_cloud = trimesh.points.PointCloud(points_blue, colors=[0, 0, 255])
        blue_mesh = pyrender.Mesh.from_points(points_blue, colors=blue_cloud.colors)
        scene.add(blue_mesh)

    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=5)

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

class PointCloudEncoder(nn.Module):
    def __init__(self, k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1 = nn.Sequential(
            nn.Conv1d(6, 64, 1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, 1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 256, 1),
            nn.AdaptiveMaxPool1d(1),
        )
        self.l2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
    
    def forward(self, x):
        x = x.permute((0, 2, 1))
        x = self.l1(x)
        x = x.squeeze()
        x = self.l2(x)
        return x

class PointCloudDecoder(nn.Module):
    def __init__(self, k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1 = nn.Sequential(
            nn.Linear(k, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, POINT_CLOUD_SIZE * 6)
        )

    def forward(self, x):
        x = self.l1(x)
        x = x.reshape((-1, POINT_CLOUD_SIZE, 6))
        return x

class PointCloudDecoder(nn.Module):
    def __init__(self, k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.u = 5
        assert POINT_CLOUD_SIZE % self.u == 0
        self.coarse_size = POINT_CLOUD_SIZE // (self.u ** 2)
        self.coarse = nn.Sequential(
            nn.Linear(k, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.coarse_size * 6)
        )
        gx, gy = torch.meshgrid((torch.arange(0, self.u), torch.arange(0, self.u)), indexing="ij")
        g = torch.concat((gx.unsqueeze(2), gy.unsqueeze(2)), dim=2).unsqueeze(0)
        g = g.repeat(self.coarse_size, 1, 1, 1)
        self.grid = torch.flatten(g, start_dim=0, end_dim=2).unsqueeze(0)
        self.fine = nn.Sequential(
            nn.Linear(k + 6 + 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 6)
        )

    def forward(self, x):
        coarse_pred = self.coarse(x)
        coarse_pred = coarse_pred.reshape((-1, self.coarse_size, 6))

        t = torch.repeat_interleave(coarse_pred, self.u ** 2, dim=1)
        xr = torch.repeat_interleave(x.unsqueeze(1), POINT_CLOUD_SIZE, dim=1)
        g = torch.repeat_interleave(self.grid, x.shape[0], dim=0).to(x.device)
        c = torch.concat((t, g, xr), dim=2)
        fine_pred = self.fine(c) + t
        
        return fine_pred, coarse_pred

class PointCloudAutoEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder = PointCloudEncoder(128)
        self.decoder = PointCloudDecoder(128)
    
    def forward(self, x):
        x_p = x.permute((0, 2, 1))
        x, _ = self.encoder(x_p)
        x = self.decoder(x)
        return x
    
class PointCloudAutoDecoder(nn.Module):
    def __init__(self, latent_size, size_train, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_size = latent_size

        self.embeddings = nn.Embedding(size_train, latent_size)
        self.decoder = PointCloudDecoder(latent_size)

    def forward(self, idx, embeddings = None):
        if embeddings is not None:
            latent = embeddings(idx)
            """
            pose = latent[:, self.latent_size:]
            latent = latent[:, :self.latent_size]
            quat = pose[:, :4]
            quat = quat / torch.norm(quat, dim=1, keepdim=True)
            qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
            R = torch.stack([
                1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw),
                2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw),
                2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)
            ], dim=-1).view(-1, 3, 3)
            t = pose[:, 4:]
            """
        else:
            latent = self.embeddings(idx)
        if len(latent.shape) == 1:
            latent = latent.unsqueeze(0)

        coarse, fine = self.decoder(latent)
        """
        if embeddings is not None:
            ct = torch.matmul(coarse[:, :, :3], R) + t.unsqueeze(1)
            cn = torch.matmul(coarse[:, :, 3:], R)
            coarse = torch.concat((ct, cn), dim=2)
            ft = torch.matmul(fine[:, :, :3], R) + t.unsqueeze(1)
            fn = torch.matmul(fine[:, :, 3:], R)
            fine = torch.concat((ft, fn), dim=2)
        """
        return coarse, fine

def chamfer_dist_slow(x1, x2):
    sub = x1.unsqueeze(1) - x2.unsqueeze(2)
    dist = (sub ** 2).sum(dim=3)
    md1, _ = torch.min(dist, dim=2)
    md2, _ = torch.min(dist, dim=1)
    cd = md1.sqrt().mean(dim=1) + md2.sqrt().mean(dim=1)
    return cd.mean()

def chamfer_dist_slow_normal(pred, target):
    p1, normalpred = pred[:, :, 0:3], pred[:, :, 3:6]
    p2, normaltarget = target[:, :, 0:3], target[:, :, 3:6]

    sub = p1.unsqueeze(1) - p2.unsqueeze(2)
    dist = (sub ** 2).sum(dim=3)
    md1, _ = torch.min(dist, dim=2) #not train tested!
    md2, sel_idx = torch.min(dist, dim=1)
    #w = np.percentile(md1.detach().cpu().numpy(), 70, axis=1, keepdims=True)
    #mask = md1 > torch.tensor(w, device=md2.device)
    #md1[mask] = 0
    cd = md1.sqrt().mean(dim=1) + md2.sqrt().mean(dim=1)

    batch_idx = torch.repeat_interleave(torch.arange(0, len(sel_idx)), sel_idx.shape[1])
    sel_idx = sel_idx.flatten()
    nt = normaltarget[batch_idx, sel_idx, :].reshape((target.shape[0], -1, 3))
    normallength = (normalpred ** 2).sum(dim=2).sqrt()
    targetlength = (nt ** 2).sum(dim=2).sqrt()
    nldir = (nt * normalpred).sum(dim=2) / (normallength * targetlength)
    normalloss = - nldir.mean(dim=1) + ((normallength - 1) ** 2).mean(dim=1)
    
    return cd.mean() + normalloss.mean() * 0.2

def save_ply(data, output_path="point_cloud.ply"):
    positions = data[:, :3]
    normals = data[:, 3:]
    #point_cloud_trimesh = trimesh.points.PointCloud(vertices=positions, normals=normals)
    point_cloud_trimesh = trimesh.Trimesh(vertices=positions, faces=np.zeros((0, 3)), vertex_normals=normals)
    point_cloud_trimesh.export(output_path)

def measure_volume(data, save_path = None, save_name = None):
    positions = data[:, :3]
    normals = data[:, 3:]
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=positions, v_normals_matrix=normals))
    #ms.save_current_mesh(f'3dvol_approx/tests/mesh{tapi}_points.ply')
    ms.generate_surface_reconstruction_screened_poisson(
        depth=8
    )
    volume = ms.get_geometric_measures()["mesh_volume"]
    if save_path:
        ms.save_current_mesh(str(save_path / f"{save_name}_{volume}.ply"))
    return volume

def evaluate(dataloader: DataLoader, model: nn.Module, metric = "chamfer"):
    model.eval()
    val_embeddings = nn.Embedding(len(dataloader.dataset), model.latent_size).to(device)
    optimizer = torch.optim.Adam(val_embeddings.parameters(), lr=0.01)
    num_epochs = 100
    errs_chamfer = []
    pred_vol = []
    direct_vol = []
    for e in range(num_epochs):
        val_losses = []
        for i, (idx, batch) in enumerate(dataloader):
            batch = batch.to(device)
            idx = idx.to(device)
            x, _ = model(idx, val_embeddings)
            
            if e == num_epochs -1:
                with torch.no_grad():
                    if metric == "chamfer":
                        loss = chamfer_dist_slow_normal(x, batch)
                        errs_chamfer.append(loss.item())
                    elif metric == "volume":
                        for j in range(len(batch)):
                            pred_vol.append(measure_volume(x[j].cpu().numpy(), str(i * batch.shape[0] + j) + "pred"))
                            direct_vol.append(measure_volume(batch[j].cpu().numpy(), str(i * batch.shape[0] + j) + "batch"))
                        print(f"Done volume for batch {i}")
            else:
                loss = chamfer_dist_slow_normal(x, batch)
                val_losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if e != num_epochs -1:
            print(f"Val epoch {e}, loss: {np.mean(val_losses)}; ", end="")
        else:
            print("")
    if metric == "chamfer":
        return np.mean(errs_chamfer)
    elif metric == "volume":
        import matplotlib.pyplot as plt
        plt.scatter(direct_vol, pred_vol)
        plt.plot((2000, 8000), (2000, 8000))
        plt.show()
        return np.mean(np.abs(np.array(pred_vol) - np.array(direct_vol)))

def train_latent_model():
    train_dataset = PlyDataset("3dvol_approx/train_ply.csv", "3dvol_approx/train_cache.pth", True)
    val_dataset = PlyDataset("3dvol_approx/val_ply.csv", "3dvol_approx/val_cache.pth", True)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = PointCloudAutoDecoder(128, len(train_loader.dataset)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, 0.5)

    best_val = None
    best_model = None
    for epoch in range(151):
        errs_train = []
        model.train()
        for idx, batch in train_loader:
            batch = batch.to(device)
            idx = idx.to(device)
            predictions, coarse_pred = model(idx)
            loss = chamfer_dist_slow_normal(predictions, batch)
            errs_train.append(loss.item())
            loss = loss + chamfer_dist_slow_normal(coarse_pred, batch)
            loss = loss + (model.embeddings.weight ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        if epoch % 30 == 0 and epoch > 0:
            err_val = evaluate(val_loader, model, "chamfer")

            if best_val is None or err_val < best_val:
                best_val = err_val
                best_model = copy.deepcopy(model).cpu()

            print(f"epoch {epoch}. Val: {err_val}, Train: {np.mean(errs_train)}")
        else:
            print(f"epoch {epoch}. Train: {np.mean(errs_train)}")

    torch.save(best_model, "3dvol_approx/model_backup_.pth")

def fit_embeddings(loader, model3d, verbose=True, epochs=300):
    embeddings = nn.Embedding(len(loader.dataset), 128).to(device)
    model3d = model3d.to(device)
    optimizer = torch.optim.Adam(embeddings.parameters(), 0.01)
    for e in range(epochs):
        losses = []
        for idx, batch in loader:
            batch = batch.to(device)
            idx = idx.to(device)
            x, _ = model3d(idx, embeddings)
            loss = chamfer_dist_slow_normal(x, batch)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if verbose:
            print(f"Epoch {e}: {np.mean(losses)}; ", end="")
    if verbose:
        print("\n")
    return embeddings

def generate_latent2volume_ds(loader: DataLoader, model3d: nn.Module, path: Path, length: int):
    """
    embeddings = fit_embeddings(loader, model3d, epochs=0).cpu()

    from sklearn.decomposition import PCA
    w = PCA(128)
    w.fit(embeddings.weight.detach().numpy())
    """

    latents = []
    volumes = []
    model3d = model3d.eval().to(device)
    with torch.no_grad():
        for i in tqdm.tqdm(range(length)):
            """
            norm = torch.randn(embeddings.embedding_dim) * w.explained_variance_
            latent = norm.unsqueeze(1) * w.components_
            latent = latent.sum(dim=0)
            """
            latent = torch.randn(128) * 1.1
            #latent =  embeddings.weight[i % len(loader.dataset)] + norm
            hemb = nn.Embedding(1, latent.shape[0])
            hemb.weight[0] = torch.nn.Parameter(latent)
            hemb = hemb.to(device)
            points, _ = model3d(torch.tensor(0, dtype=torch.int32, device=device), hemb)
            volume = measure_volume(points[0].cpu().numpy())

            latents.append(latent.tolist())
            volumes.append(volume)

    df = pd.DataFrame({"volume": volumes, "latents": latents})
    df.to_csv(path, index=False)

def create_latent2volume_ds(loader: DataLoader, model3d: nn.Module, path: Path, ):
    embeddings = fit_embeddings(loader, model3d, epochs=150).cpu()

    ds_df: pd.DataFrame = loader.dataset.df.copy()

    ds_df["latents"] = embeddings.weight.tolist()
    ds_df.to_csv(path, index=False)

def direct_latent2volume_ds():
    train_dataset = PlyDataset("3dvol_approx/train_ply.csv", "3dvol_approx/train_cache.pth", True)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataset = PlyDataset("3dvol_approx/val_ply.csv", "3dvol_approx/val_cache.pth", True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    test_dataset = PlyDataset("3dvol_approx/test_ply.csv", "3dvol_approx/test_cache.pth", True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    model3d = torch.load("3dvol_approx/model_backup.pth")
    create_latent2volume_ds(train_loader, model3d, "3dvol_approx/latent2volume_train.csv")
    create_latent2volume_ds(val_loader, model3d, "3dvol_approx/latent2volume_val.csv")
    create_latent2volume_ds(test_loader, model3d, "3dvol_approx/latent2volume_test.csv")

class Latent2VolumeDs(Dataset):
    def __init__(self, path):
        super().__init__()
        self.df = pd.read_csv(path)
        self.df["latents"] = self.df["latents"].apply(ast.literal_eval)

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        latent = self.df.loc[index, "latents"]
        volume = self.df.loc[index, "volume"]
        return torch.tensor(latent), torch.tensor(Latent2VolumeDs.norm_vol(volume), dtype=torch.float32)
    
    @staticmethod
    def norm_vol(v):
        return (v - 4763) / 1226
    
    @staticmethod
    def unorm_vol(v):
        return v * 1226 + 4763


def train_latent2volume():
    train_dataset = Latent2VolumeDs("3dvol_approx/latent2volume_train_gen.csv")
    val_dataset = Latent2VolumeDs("3dvol_approx/latent2volume_val.csv")
    test_dataset = Latent2VolumeDs("3dvol_approx/latent2volume_test.csv")
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    volume_model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 1)
    )

    def evaluate(loader, model, scatter = False):
        model = model.eval()
        with torch.no_grad():
            preds = []
            vols = []
            for latent, volume in loader:
                x = model(latent).squeeze()
                preds.append(x)
                vols.append(volume)
            preds = Latent2VolumeDs.unorm_vol(torch.concat(preds))
            vols = Latent2VolumeDs.unorm_vol(torch.concat(vols))

            if scatter:
                import matplotlib.pyplot as plt
                plt.plot((2000, 8000), (2000, 8000))
                plt.scatter(vols, preds)
                plt.show()
            
            print(f"MAE: {np.mean(np.abs(np.array(preds) - np.array(vols)))}")
    
    optimizer = torch.optim.Adam(volume_model.parameters(), lr=0.0005)
    for epoch in range(200):
        losses = []
        volume_model = volume_model.train()
        for latent, volume in train_loader:
            x = volume_model(latent).squeeze()
            loss = ((x - volume) ** 2).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        print(f"Epoch {epoch}, {np.mean(losses)}")
        evaluate(val_loader, volume_model)
    evaluate(test_loader, volume_model, True)

def h():
    class QuickPclLoader(Dataset):
        def __init__(self, path: Path):
            super().__init__()
            self.data = []
            for p in path.glob("*.ply"):
                p = trimesh.load_mesh(p)
                p = p.metadata["_ply_raw"]["vertex"]["data"]
                d = np.zeros((len(p), 6))
                for i in range(len(p)):
                    h = p[i]
                    for j in range(len(h)):
                        d[i, j] = h[j]
                self.data.append(torch.tensor(d))
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, index):
            return index, self.data[index]
        
    ds = QuickPclLoader(Path(r"C:\Users\Admin\Desktop\master_thesis\volume_prediction_fip\3dvol_approx\real_tests"))
    #ds = PlyDataset(r"C:\Users\Admin\Desktop\master_thesis\volume_prediction_fip\3dvol_approx\fake_tests\partial.csv")
    loader = DataLoader(ds, 4)
    model = torch.load("3dvol_approx/model_backup.pth")
    embeddings = fit_embeddings(loader, model, epochs=3000).cpu()
    model = model.eval().cpu()
    with torch.no_grad():
        for idx, batch in loader:
            pred, _ = model(idx, embeddings)
            for i in range(len(batch)):
                visualize_point_clouds(batch[i, :, :3].numpy(), pred[i, :, :3].numpy())
                #measure_volume(pred[i], Path(r"C:\Users\Admin\Desktop\master_thesis\volume_prediction_fip\3dvol_approx\fake_tests\export_fit_partial"), f"{i}")
                pass

if __name__ == "__main__":
    #train_latent_model()
    entire_dataset = PlyDataset("3dvol_approx/entire_dataset_ply.csv", "3dvol_approx/entire_dataset_cache.pth", True)
    train_loader = DataLoader(entire_dataset, batch_size=16, shuffle=False)
    model3d = torch.load("3dvol_approx/model_backup.pth")
    emb = fit_embeddings(train_loader, model3d, epochs=100)
    df_out = entire_dataset.df
    df_out["latent"] = emb.weight.detach().cpu().tolist()
    df_out.to_csv("3dvol_approx/train_latent_out.csv", index=False)
    #generate_latent2volume_ds(train_loader, model3d, "3dvol_approx/latent2volume_train_gen2.csv", 50000)
        