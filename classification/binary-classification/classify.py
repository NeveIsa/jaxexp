import numpy as np
import jax.numpy as jnp
import jax.nn as jnn
from jax import grad, jit
from jax.tree_util import tree_map
from fire import Fire
import mnist
from tqdm import tqdm


def loadmnist():
    X = mnist.test_images()
    y = mnist.test_labels()

    zeros = np.where(y == 0)[0].tolist()
    ones = np.where(y == 1)[0].tolist()

    zeros_and_ones_train = zeros[:-100] + ones[:-100]
    zeros_and_ones_test = zeros[-100:] + ones[-100:]

    Xtrain, Xtest = X[zeros_and_ones_train], X[zeros_and_ones_test]
    ytrain, ytest = y[zeros_and_ones_train], y[zeros_and_ones_test]

    Xtrain = np.vstack([_.flatten() for _ in Xtrain]).T
    Xtest = np.vstack([_.flatten() for _ in Xtest]).T

    # normalize -> this is critical for convergence of the network
    Xtrain = Xtrain / 255
    Xtest = Xtest / 255

    return Xtrain, ytrain.reshape(1, -1), Xtest, ytest.reshape(1, -1)


# @jit
def neuralnet(params, X, activation):
    for p in reversed(params):
        X = activation(p @ X)
    return X


# @jit
def lossfn(params, X, y, activation):
    yhat = neuralnet(params, X, activation)
    loss = jnp.linalg.norm(y - yhat)
    return loss


def train(params_init, X, y, activation, lr, iters):
    params = params_init

    dlossfn = grad(lossfn)
    # dlossfn = jit(grad(lossfn))

    pbar = tqdm(range(iters))
    for p in pbar:
        gradient = dlossfn(params, X, y, activation)
        params = tree_map(lambda a, b: a - lr * b, params, gradient)

        if p % 10 == 0:
            loss = lossfn(params, X, y, activation)
            pbar.set_postfix({"loss": loss})

    return params


def main(iters=300, lr=0.1, act="sigmoid"):
    actfn = eval(f"jnn.{act}")

    Xtrain, ytrain, Xtest, ytest = loadmnist()

    # ytrain = 2*ytrain - 1

    scale = 1
    params = [
        np.random.randn(1, 10),
        np.random.randn(10, 100),
        np.random.randn(100, 100),
        np.random.randn(100, 100),
        np.random.randn(100, 100),
        np.random.randn(100, 28 * 28),
    ]
    params = [p * scale for p in params]

    # loss = lossfn(params, Xtest, ytest, activation=actfn)
    # print(loss)

    params = train(params, Xtrain, ytrain, activation=actfn, lr=lr, iters=iters)

    yhat = neuralnet(params, Xtest, activation=actfn)
    yhat = (yhat > 0.5) * 1

    test_err = (1 * (yhat == ytest)).sum() / ytest.shape[1]
    print(f"Test Accuracy  -> {100*test_err:2f}%")

    # print(yhat)
    # print(ytest)


if __name__ == "__main__":
    Fire(main)
