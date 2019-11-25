import torch
from model import *
import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import chain

if __name__ == "__main__":
    x, _ = sklearn.datasets.make_swiss_roll(10000)
    x = x.astype(np.float32)
    print(x.shape)
    y = (x[:, 1] < 10).astype(np.int32)
    print(np.sum(y), y.shape)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x[:, 0], x[:, 1], x[:, 2])
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.show()

    decoder_x = DecoderX(4, 3)
    encoder_w = EncoderW(4, 2)
    encoder_z = EncoderZ(3, 2)
    decoder_y = DecoderY(2)

    optimizer = torch.optim.Adam(chain(decoder_x.parameters(), 
        encoder_w.parameters(), 
        encoder_z.parameters(), 
        decoder_y.parameters()), lr=1e-3)
    
    trainloader = torch.utils.data.DataLoader(list(zip(x, y)), shuffle=True, batch_size=100)

    for x, y in trainloader:
        mu_z, logvar_z, z = encoder_z(x)
        mu_w, logvar_w, w = encoder_w(x, y.unsqueeze(-1).float())

        mu_x, logvar_x, pred_x = decoder_x(w, z)
        pred_y = decoder_y(z)[:, 1]
        print(pred_y.shape)