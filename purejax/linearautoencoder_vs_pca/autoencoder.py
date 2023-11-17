from fire import Fire
from jax import grad, jit
import jax.numpy as jnp
import jax.nn as jnn
from jax.tree_util import tree_map
import mnist
import numpy as np
import random
from tqdm import tqdm
from sklearn.decomposition import PCA
import seaborn as sns
from scipy.spatial import procrustes

from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (15,5)

# from celluloid import Camera

def load_mnist(train=1000, test=100):
    train_images, train_labels = mnist.train_images(), mnist.train_labels()
    test_images, test_labels = mnist.test_images(), mnist.test_labels()

    zeros = np.where(train_labels==0)
    ones = np.where(train_labels==1)
    # print(train_images.shape);exit()

    zerosandones = np.hstack([zeros, ones]).tolist()[0]
    train_labels = train_labels[zerosandones]
    train_images = train_images[zerosandones]

    # print(train_images.shape);exit()
        
    n_train = train_images.shape[0]
    n_test = test_images.shape[0]

    train_images = train_images.reshape(n_train, -1)
    test_images = test_images.reshape(n_test, -1)

    train_indices = random.sample(range(n_train), 1000)
    test_indices = random.sample(range(n_test), 100)

    train_images = train_images[train_indices, :]
    test_images = test_images[test_indices, :]

    train_labels = train_labels[train_indices]
    test_labels = test_labels[test_indices]
    
    # print(train_images.shape, test_images.shape)

    return train_images.T, train_labels.T, test_images.T, test_labels.T

def encoder(weights, X):
    for w in weights:
        X = w @ X
    return X

def decoder(weights, X):
    for w in weights:
        X = w @ X
    return X

def autoencoder(eweights, dweights, X):
    X = encoder(eweights, X)
    X = decoder(dweights, X)
    return X

@jit
def mseloss(weights, X):
    if len(weights)==1:
        ew = weights[0]
        dw = [w.T for w in weights[0]]
        Xhat = autoencoder(ew,dw,X) 
    else:
        Xhat = autoencoder(weights[0], weights[1], X)

    n = X.shape[1]
    return (jnp.linalg.norm(X - Xhat)**2)/n

def train(iters=100, lr=1e-8, latentdim=2):
    xtrain, ytrain, xtest, ytest = load_mnist()
    xdim = xtrain.shape[0]

    xtrain = ( xtrain.T - xtrain.T.mean(axis=0) ).T

    encw = [np.random.rand(latentdim,xdim)/10000]
    decw = [np.random.rand(xdim,latentdim)/10000]
    
    # w = [encw,decw]
    w = [encw]

    if len(w)==2:
        print("before:", np.linalg.norm(w[0][0] - w[1][0].T)/(np.linalg.norm(w[0][0])+np.linalg.norm(w[1][0])) )

    dloss = jit(grad(mseloss, argnums=0))
    
    # train
    pbar = tqdm(range(iters))
    for i in pbar:
        dw = dloss(w,xtrain)
        if i%50==0:
            loss = mseloss(w,xtrain)
            lr = lr*1.01
        w = tree_map(lambda x,y: x - lr*y, w, dw)
        # print(dw)
        pbar.set_postfix({"loss":loss})
        # input()  

    if len(w)==1:
        xestimate = autoencoder(w[0],[ _w.T for _w in w[0] ],xtrain.copy())
    else:
        xestimate = autoencoder(w[0],w[1],xtrain.copy())
        
    print(  "autoencoder:",(np.linalg.norm(xestimate)**2)/(np.linalg.norm(xtrain)**2)   )
    
    pca = PCA(n_components=latentdim)
    xpca = pca.fit_transform(xtrain.T)
    print("pca:",sum(pca.explained_variance_ratio_))
    # print(xpca.shape)

    plt.subplot(1,3,1)
    sns.scatterplot(x=xpca[:,0], y=xpca[:,1], hue=ytrain.tolist(), palette="PRGn")
    plt.title(f"PCA: {latentdim} dim")

    xencode = encoder(w[0],xtrain).T
    plt.subplot(1,3,2)
    sns.scatterplot(x=xencode[:,0], y=xencode[:,1], hue=ytrain.tolist(), palette="seismic")
    plt.title(f"Autoencoder: {latentdim} dim")

    # print(xencode.shape)

    x1,x2,match_error = procrustes(xpca,xencode)
    print("match_error:", match_error)    

    plt.subplot(1,3,3)
    sns.scatterplot(x=x1[:,0], y=x1[:,1], marker="o", hue=ytrain.tolist(), palette="PRGn")
    sns.scatterplot(x=x2[:,0], y=x2[:,1], marker="2", hue=ytrain.tolist(), palette='seismic')

    plt.title(f"Aligned latent space of PCA and linear Autoencoder")
    plt.suptitle(f"Autoencoder GD iters = {iters}")
    plt.savefig(f"plots/iters={iters}.png")    

    if len(w)==2:
        print("after:", np.linalg.norm(w[0][0] - w[1][0].T)/(np.linalg.norm(w[0][0])+np.linalg.norm(w[1][0])) )

    print(  w[0][0] @ w[0][0].T   )
    
if __name__=="__main__":
    np.random.seed(108)
    random.seed(108)
    Fire(train)
