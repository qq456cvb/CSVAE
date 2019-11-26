import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderX(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_out = dim_out
        self.fcs = nn.Sequential(
            nn.Linear(dim_in, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * dim_out)
        )

    def forward(self, w, z):
        out = self.fcs(torch.cat((w, z), dim=-1))
        mu = out[:, :self.dim_out]
        logvar = out[:, self.dim_out:]  # decorrelated gaussian
        sample = torch.randn_like(mu) * torch.exp(0.5 * logvar) + mu
        return mu, logvar, sample


class EncoderW(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_out = dim_out
        self.fcs = nn.Sequential(
            nn.Linear(dim_in, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * dim_out)
        )

    def forward(self, x, y):
        out = self.fcs(torch.cat((x, y), dim=-1))
        mu = out[:, :self.dim_out]
        logvar = out[:, self.dim_out:]  # decorrelated gaussian
        sample = torch.randn_like(mu) * torch.exp(0.5 * logvar) + mu
        return mu, logvar, sample


class EncoderZ(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_out = dim_out
        self.fcs = nn.Sequential(
            nn.Linear(dim_in, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * dim_out)
        )

    def forward(self, x):
        out = self.fcs(x)
        mu = out[:, :self.dim_out]
        logvar = out[:, self.dim_out:]  # decorrelated gaussian
        sample = torch.randn_like(mu) * torch.exp(0.5 * logvar) + mu
        return mu, logvar, sample


class DecoderY(nn.Module):
    def __init__(self, dim_in, n_class=2):
        super().__init__()
        self.fcs = nn.Sequential(
            nn.Linear(dim_in, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_class)
        )

    def forward(self, z):
        out = F.softmax(self.fcs(z), dim=-1)
        return out
