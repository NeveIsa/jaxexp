import numpy as np
import jax.nn as jnn
import jax.numpy as jnp
from jax import grad

## LAYERS ##
class Linear:
    def __init__(self, insize, outsize, bias=True):
        self.w = np.random.randn(outsize, insize)
        self.b = np.random.randn(outsize,1)
        self.params = [self.w,self.b]

    def __call__(self, params, X):
        w, b = params
        return w@X + b

class Relu:
    def __init__(self):
        self.params = None
    def __call__(self, params, X):
        return jnn.relu(X)

class Sigmoid:
    def __init__(self):
        self.params = None
    def __call__(self, params, X):
        return jnn.sigmoid(X)

class Flatten:
    def __init__(self):
        self.params = None
    def __call__(self, params, X):
        return X.flatten()
    

## NNET ##
class NeuralNet:
    def __init__(self, layers):
        self.layers = layers
        self.params = [ l.params for l in layers ]
    
    def __call__(self, params, X):
        for layer, p in zip(self.layers, params):
            X = layer(p, X)
        return X


if __name__=="__main__":
    layers = [Linear(3,2),Relu(),Linear(2,1),Relu(), Flatten()]
    nn = NeuralNet(layers)
    nnparams = nn.params

    x = np.random.rand(3,1)
    y = np.zeros((1,1))

    lossfn = lambda params, x, y : jnp.linalg.norm(nn(params,x)-y)**2
    dlossfn = grad(lossfn)

    loss = lossfn(nnparams,x,y)
    dloss = dlossfn(nnparams,x,y)
    print(f"loss:{loss}")
    print(f"dloss:{dloss}")

    
