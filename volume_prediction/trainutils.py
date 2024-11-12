import sys
sys.path.append(r"C:\Users\Admin\Desktop\master_thesis\volume_prediction_fip")

from matplotlib import pyplot as plt
import torch
import torch.utils.data
from torch import nn
from IPython.display import clear_output

class DefaultEvaluator():
    def __init__(self, dataset_val, metric, aggregate = None, device = 'cuda' if torch.cuda.is_available() else 'cpu',
                 plot_val = True, plot_loss = True):
        self.dataset = dataset_val
        self.metric = metric
        self.aggregate = aggregate
        self.device = device
        self.plot_val = plot_val
        self.plot_loss = plot_loss

        self.loss_history = []
        self.val_history = []

    def evaluate(self, model):
        with torch.no_grad():
            model.eval()
            dl = torch.utils.data.DataLoader(self.dataset, 500, pin_memory=True, pin_memory_device=self.device)

            valuations = []
            for img, vol in dl:
                vol = vol.unsqueeze(1)
                img = img.to(self.device)
                vol = vol.to(self.device)
                pred = model(img)
                v = self.metric(pred, vol)
                valuations.extend(v)

            if self.aggregate:
                valuations = self.aggregate(valuations)

            return valuations
        
    def __call__(self, model, loss = None):
        self.loss_history.append(loss)
        v = self.evaluate(model)
        self.val_history.append(v)
        if self.plot_loss or self.plot_val:
            clear_output(True)
            if self.plot_loss:
                plt.plot(self.loss_history, label=f"Train-loss {self.loss_history[-1]}")
                plt.legend()
                plt.show()
            if self.plot_val:
                plt.plot(self.val_history, label=f"Val {self.val_history[-1]}")
                plt.legend()
                plt.show()

def metric_ae(pred, target):
    return torch.abs(pred - target)

def metric_sae(pred, target):
    return pred - target

def aggregate_mean(v):
    return torch.mean(torch.tensor(v))

def train(dataset_train, model: nn.Module, optimizer, scheduler, loss_fn, val_fn,
           epochs = 100, val_step=5, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    dl = torch.utils.data.DataLoader(dataset_train, 250, True, pin_memory=True, pin_memory_device=device)
    #scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=epochs)

    model = model.to(device)
    model.train()

    for epoch in range(epochs):
        loss_glob = 0
        num_items = 0
        for img, vol in dl:
            vol = vol.unsqueeze(1)
            img = img.to(device)
            pred = model(img)
            vol = vol.to(device)
            loss = loss_fn(pred, vol)
            loss.backward()
            optimizer.step()
            loss_glob += loss.item()
            num_items += pred.shape[0]
        if scheduler:
            scheduler.step()
        
        if (epoch > 0 and epoch % val_step == 0) or epoch == epochs - 1 or epoch == 0:
            loss_avg = loss_glob / num_items
            val_fn(model, loss_avg)

            model.train()

def save(model, optim, path):
    torch.save({
        "model": model.state_dict(),
        "optim": optim.state_dict()
    }, path)

def load(model: nn.Module, optim, path, strict=True):
    j = torch.load(path)
    model.load_state_dict(j["model"], strict=strict)
    optim.load_state_dict(j["optim"])

def exec_and_ignore(fun):
    try:
        fun()
    except BaseException as e:
        import traceback
        print(traceback.print_exception(e))
        pass
