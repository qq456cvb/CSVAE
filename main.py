import torch
import torch.utils.data
from model import *
import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import chain
from tqdm import tqdm, trange


def KL(mu1, logvar1, mu2, logvar2):
    std1 = torch.exp(0.5 * logvar1)
    std2 = torch.exp(0.5 * logvar2)
    return torch.sum(torch.log(std2) - torch.log(std1) + 0.5 * (torch.exp(logvar1) + (mu1 - mu2) ** 2) / torch.exp(logvar2) - 0.5, dim=-1)


def project(ax, ys, attr, title=None):
    # scatter plot
    # plt.figure()
    colors = np.zeros((ys.shape[0], 3))
    colors[ys == 1] = np.array((1., 0, 0))
    colors[ys == 0] = np.array((0, 0, 1.))
    if title is not None:
        ax.set_title(title)
    ax.scatter(attr[:, 0], attr[:, 1], s=1, c=colors, alpha=0.3)

if __name__ == "__main__":
    xs, _ = sklearn.datasets.make_swiss_roll(10000)
    xs = xs.astype(np.float32)
    print(xs.shape)
    ys = (xs[:, 1] < 10).astype(np.int32)
    print(np.sum(ys), ys.shape)

    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.scatter(xs[:, 0], xs[:, 1], xs[:, 2])
    ax.set_title('origin')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # plt.show()


    decoder_x = DecoderX(4, 3)
    encoder_w = EncoderW(4, 2)
    encoder_z = EncoderZ(3, 2)
    decoder_y = DecoderY(2)

    optimizer1 = torch.optim.Adam(chain(decoder_x.parameters(), 
        encoder_w.parameters(), 
        encoder_z.parameters()), lr=1e-3)
    optimizer2 = torch.optim.Adam(decoder_y.parameters(), lr=1e-3)
    
    trainloader = torch.utils.data.DataLoader(list(zip(xs, ys)), shuffle=True, batch_size=256)

    with trange(30) as t:
        for i in t:
            t.set_description('Epoch %d' % i)
            for x, y in trainloader:
                
                mu_z, logvar_z, z = encoder_z(x)
                mu_w, logvar_w, w = encoder_w(x, y.unsqueeze(-1).float())

                mu_x, logvar_x, pred_x = decoder_x(w, z)
                pred_y = decoder_y(z)

                kl1 = KL(mu_w, logvar_w, torch.zeros_like(mu_w), torch.ones_like(logvar_w) * np.log(0.01))
                kl0 = KL(mu_w, logvar_w, torch.ones_like(mu_w) * 3., torch.zeros_like(logvar_w))

                optimizer1.zero_grad()
                loss1 = (20. * torch.sum((x - mu_x) ** 2, -1)
                    + 1. * torch.where(y == 1, kl1, kl0)
                    + 0.2 * KL(mu_z, logvar_z, torch.zeros_like(mu_z), torch.zeros_like(logvar_z))
                    + 1000. * torch.sum(pred_y * torch.log(pred_y), -1)).sum()  # maximize entropy, enforce uniform distribution
                loss1.backward(retain_graph=True)
                optimizer1.step()
                
                optimizer2.zero_grad()
                loss2 = (100. * torch.where(y == 1, -torch.log(pred_y[:, 1]), -torch.log(pred_y[:, 0]))).sum()
                loss2.backward()
                optimizer2.step()

                loss = loss1 + loss2
            t.set_postfix(loss=loss.item(), y_max=pred_y.max().item(), y_min=pred_y.min().item())
            # print(loss.item())

    # reconstruction
    with torch.no_grad():
        zs, _, _ = encoder_z(torch.from_numpy(xs))
        ws, _, _ = encoder_w(torch.from_numpy(xs), torch.from_numpy(ys).unsqueeze(-1).float())
        pred_x, _, _ = decoder_x(ws, zs)
        pred_x = pred_x.cpu().numpy()


    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.scatter(pred_x[:, 0], pred_x[:, 1], pred_x[:, 2])
    ax.set_title('reconstruction')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    project(fig.add_subplot(2, 2, 3), ys, ws, 'w projection')
    project(fig.add_subplot(2, 2, 4), ys, zs, 'z projection')

    plt.show()