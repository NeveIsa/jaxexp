import sys
sys.path.append("../../jaxhelpers")
from load01 import loadmnist
import neuralnetslib as nnlib

from jaxopt import GradientDescent
import jax.numpy as jnp

def lossfn(params, X, y):
    yhat = nn(params,X)
    loss = jnp.linalg.norm(y - yhat)**2
    return loss

Xtrain, ytrain, Xtest, ytest = loadmnist()

layers = [
    nnlib.Linear(28 * 28, 100),
    nnlib.Sigmoid(),
    nnlib.Linear(100, 100),
    nnlib.Sigmoid(),
    nnlib.Linear(100, 50),
    nnlib.Sigmoid(),
    nnlib.Linear(50, 10),
    nnlib.Sigmoid(),
    nnlib.Linear(10, 1),
    nnlib.Sigmoid(),
]
nn = nnlib.NeuralNet(layers)
init_params = nn.params

solver = GradientDescent(lossfn, maxiter=300, verbose=True)
res = solver.run(init_params, Xtrain, ytrain)

yhat = nn(res.params, Xtest)
yhat = (yhat > 0.5) * 1
test_err = (1 * (yhat == ytest)).sum() / ytest.shape[1]
print(f"Test Accuracy (jaxopt) -> {100*test_err:2f}%")

