import torch
from torch import nn
import torch.optim as optim

import numpy as np
from matplotlib import pyplot as plt
from pixel_from_coordinates import load_img

from tqdm import tqdm
from fire import Fire


# https://pytorch.org/docs/stable/generated/torch.func.grad.html
def main(iters=1000, n_fourier_features=8, lr=1e-2, schedulefn="np.log"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, test_data = load_img()
    _,image = test_data
    plt.imshow(image)
    plt.title("original")
    plt.savefig("orig.png")
    X = [[x,y] for x in range(image.shape[0]) for y in range(image.shape[1])]
    X = np.vstack(X, dtype=float).T/image.shape[0]

    # fourier featuers
    if n_fourier_features:
        X = 2*np.pi*X 
        X = np.random.randn(n_fourier_features,2) @ X * 1 # scale=1
        X = np.concatenate([np.sin(X), np.cos(X)])
    X=torch.tensor(X,dtype=torch.float32).to(device)

    y = [image[x,y] for x in range(image.shape[0]) for y in range(image.shape[1])]
    y = np.vstack(y, dtype=float).T
    y = torch.tensor(y,dtype=torch.float32).to(device)

    model = nn.Sequential(
        nn.Linear(X.shape[0], 256),
        nn.ReLU(),
        nn.Linear(256,256),
        nn.ReLU(),
        nn.Linear(256,256),
        nn.ReLU(),
        nn.Linear(256,256),
        nn.ReLU(),
        nn.Linear(256,256),
        nn.ReLU(),
        nn.Linear(256,256),
        nn.ReLU(),
        nn.Linear(256,y.shape[0]),
        nn.Sigmoid()
    ).to(device)

    torch.set_float32_matmul_precision("high")
    model = torch.compile(model)

    lossfn = nn.MSELoss()

    # create your optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr/1000)

    pbar = tqdm(range(iters))
    for i in pbar:
        optimizer.zero_grad() 

        # https://discuss.pytorch.org/t/runtimeerror-mat1-and-mat2-must-have-the-same-dtype/166759/4  
        yhat = model(X.T).T
        loss = lossfn(y,yhat)
        loss.backward()
        optimizer.step()

        pbar.set_postfix({"loss":loss.item()})

    yhat = yhat.T.reshape(*image.shape).cpu().detach().numpy()
    print(yhat.shape)
    plt.imshow(yhat)
    plt.title(f"nff={n_fourier_features} | iters={iters}")
    plt.savefig(f"pytorch_plots/nff={n_fourier_features}.png")

if __name__=="__main__":
    Fire(main)