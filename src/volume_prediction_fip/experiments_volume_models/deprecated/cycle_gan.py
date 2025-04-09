import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
import itertools

# https://arxiv.org/pdf/1703.10593
# https://arxiv.org/pdf/1904.00069

class Mapfun(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        return self.l(x)
    
class Discriminator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        self.l(x)

    def forward_loss_discriminator(self, real_samples, fake_samples):
        real_preds = self.forward(real_samples)
        fake_preds = self.forward(fake_samples)

        lreal = F.mse_loss(real_preds, 1)
        lfake = F.mse_loss(fake_preds, 0)

        return (lreal + lfake) / 2
    
    def forward_loss_generator(self, fake_samples):
        fake_preds = self.forward(fake_samples)
        return F.mse_loss(fake_preds, 1)
    
def train_gan():
    dataset_complete = object()
    dataset_partial = object()
    dl_complete = DataLoader(dataset_complete, 32, True)
    dl_partial = DataLoader(dataset_partial, 32, True)

    complete2partial = Mapfun()
    partial2complete = Mapfun()
    discrimatorcomplete = Discriminator()
    discrimatorpartial = Discriminator()

    optim_discriminator = Adam(itertools.chain(discrimatorcomplete.parameters(), discrimatorpartial.parameters()))
    optim_generator = Adam(itertools.chain(complete2partial.parameters(), partial2complete.parameters()))

    for epoch in range(100):
        for batch_complete, batch_partial in zip(dl_complete, dl_partial):

            pred_partial = complete2partial(batch_complete)
            pred_complete = partial2complete(batch_partial)

            loss_discriminator = discrimatorcomplete.forward_loss_discriminator(pred_complete, pred_partial) + \
                discrimatorpartial.forward_loss_discriminator(pred_partial, pred_complete)
            loss_discriminator.backward()
            optim_discriminator.step()
            optim_discriminator.zero_grad()

            loss_cylce = F.l1_loss(partial2complete(pred_partial), batch_complete) + F.l1_loss(complete2partial(pred_complete), batch_partial)
            loss_gen = discrimatorcomplete.forward_loss_generator(pred_partial) + discrimatorpartial.forward_loss_discriminator(pred_complete)
            loss = loss_cylce * 10 + loss_gen
            loss.backward()
            optim_generator.step()
            optim_generator.zero_grad()

            print(loss.item())