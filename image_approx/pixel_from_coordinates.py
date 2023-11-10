from jax import grad, jit
import jax.numpy as jnp
import jax.nn as jnn
from jax.tree_util import tree_map

import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import mnist
from fire import Fire
import imageio.v2 as imageio

########## LAYERS ##########
class Linear:
    def __init__(self, indim, outdim, bias=True, scale=1):
        # https://www.deeplearning.ai/ai-notes/initialization/index.html#III
        if scale==1:
            self.W = np.random.normal(loc=0, scale=2/indim ,size=(outdim, indim))
        else:
            # self.W = np.random.normal(loc=0, scale=scale,size=(outdim, indim))
            self.W = np.random.randn(outdim, indim)*scale


        self.bias = bias

        if bias:
            self.b = np.zeros((outdim, 1))

    def params(self):
        return self.W, self.b if self.bias else self.W

    def apply(self, params, X):
    
        assert len(params)>0
        
        if len(params)==1:
            W = params
            return W @ X
        else:
            W,b = params
            return W @ X + b

    def __str__(self):
        return f"Linear({self.W.shape[1]},{self.W.shape[0]},bias={self.bias})"


class Sigmoid:
    def params(self):
        return []

    def apply(self, params, X):
        return jnn.sigmoid(X)

    def __str__(self):
        return f"Sigmoid"


class Relu:
    def params(self):
        return []

    def apply(self, params, X):
        return jnn.relu(X)

    def __str__(self):
        return f"Relu"


class Selu:
    def params(self):
        return []

    def apply(self, params, X):
        return jnn.selu(X)
    def __str__(self):
        return f"Selu"


class LeakyRelu:
    def params(self):
        return []

    def apply(self, params, X):
        return jnn.leaky_relu(X)

    def __str__(self):
        return f"LeakyRelu"


class Tanh:
    def params(self):
        return []

    def apply(self, params, X):
        return jnn.tanh(X)

    def __str__(self):
        return f"Tanh"
        
########## LAYERS ##########
class NeuralNet:
    def __init__(self, layers):
        self.layers = layers

    def params(self):
        return [ layer.params() for layer in self.layers ]

    def apply(self, params, X):
        for layer,param in zip(self.layers,params):
            X = layer.apply(param,X)
        return X

    def __str__(self):
        txt = "\n---------\n"
        txt += "NeuralNet"
        txt += "\n---------\n"
        txt += "\n".join(   [ l.__str__() for l in self.layers ]   )
        txt += "\n---------\n"
        return txt

class GD:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        self.N = np.prod(self.y.shape)
    
    def lossfn(self, params):
        yhat = self.model.apply(params, self.X)
        err = yhat - self.y
        return (jnp.linalg.norm(err)**2)/self.N
    
    def train(self, params, iters, lr, schedule_fn=np.log):
        lossfngrad = jit(grad(self.lossfn, argnums=0)) 
        # since we are already using self.loss, argnums is 0 which refers to params

        # loss = self.lossfn(params)
        # print(loss)
        # params = tree_map(lambda x: x*2, params)
        # loss = self.lossfn(params)
        # print(loss)
            
        # exit()
        
        pbar = tqdm(range(iters), colour="red", leave=False)
        for p in pbar:
            params_grad = lossfngrad(params)
            params = tree_map(lambda x, y: x -  y * lr * schedule_fn(p), params, params_grad)
            if p%25==0: 
                __loss = self.lossfn(params)
                pbar.set_postfix({"loss": __loss})
                # print(params_grad)

        return params

        
################ MNIST #############
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


def load_img():
    # Download image, take a square crop from the center
    # image_url = 'https://live.staticflickr.com/7492/15677707699_d9d67acf9d_b.jpg'
    image_url = 'image.jpg'

    img = imageio.imread(image_url)[..., :3] / 255.
    c = [img.shape[0]//2, img.shape[1]//2]
    r = 256
    img = img[c[0]-r:c[0]+r, c[1]-r:c[1]+r]
    
    # plt.imshow(img)
    # plt.show()
    
    # Create input pixel coordinates in the unit square
    coords = np.linspace(0, 1, img.shape[0], endpoint=False)
    x_test = np.stack(np.meshgrid(coords, coords), -1)
    test_data = [x_test, img]
    train_data = [x_test[::2, ::2], img[::2, ::2]]

    return train_data, test_data

################ MNIST #############


def main(iters=1000, n_fourier_features=8, lr=1e1, schedulefn="np.log"):
    random.seed(108)


    # xtrain,ytrain,_,_ = load_mnist()
    # # print(X.shape)

    # image = xtrain[:,0].reshape(28,28) # we just need one image
    # image = image/255
    # print(image.min())
    
    # # plot original
    # plt.imshow(image)
    # plt.savefig("orig.png")
    
    # X = [[x,y] for x in range(28) for y in range(28)]
    # y = [ image[x,y] for x in range(28) for y in range(28)]
    
    # X = np.vstack(X, dtype=float).T/28
    # y = np.vstack(y, dtype=float)

    # exit()


    train_data, test_data = load_img()
    _,image = test_data
    plt.imshow(image)
    plt.title("original")
    plt.savefig("orig.png")
    X = [[x,y] for x in range(image.shape[0]) for y in range(image.shape[1])]
    X = np.vstack(X, dtype=float).T/image.shape[0]

    # fourier featuers
    X = 2*np.pi*X 
    X = np.random.randn(n_fourier_features,2) @ X * 1 # scale=1
    X = np.concatenate([jnp.sin(X), jnp.cos(X)])
    # print(X.shape);exit()


    y = [image[x,y] for x in range(image.shape[0]) for y in range(image.shape[1])]
    y = np.vstack(y, dtype=float).T
    # print(y.shape, X.shape);exit()

    scale=1e-1
    nnmodel = NeuralNet(
        [
            Linear(X.shape[0], 256,scale=scale),
            Relu(),
            Linear(256, 256, scale=scale),
            Relu(),
            Linear(256, 256, scale=scale),
            Relu(),
            Linear(256, 256, scale=scale),
            Relu(),
            Linear(256, 256, scale=scale),
            Relu(),
            Linear(256, 256, scale=scale),
            Relu(),
            Linear(256,y.shape[0], scale=scale),
            Sigmoid(),
            # Tanh()
        ]
    )
    
    print(nnmodel)

    nnparams = nnmodel.params()

    gd = GD(nnmodel, X, y)
    nnparams = gd.train(nnparams,iters=iters, lr=lr, schedule_fn=eval(schedulefn))    
    
    yhat = nnmodel.apply(nnparams,X).T.reshape(*image.shape)
    plt.imshow(yhat)
    # plt.title(f"nff={n_fourier_features}")
    plt.title(f"nff={n_fourier_features} | iters={iters}")

    plt.savefig(f"plots/nff={n_fourier_features}.png")


if __name__ == "__main__":
    Fire(main)
