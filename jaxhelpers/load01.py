import numpy as np
import mnist

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
    # https://stats.stackexchange.com/questions/421927/neural-networks-input-data-normalization-and-centering
    # https://stats.stackexchange.com/questions/420231/effect-of-rescaling-of-inputs-on-loss-for-a-simple-neural-network/420330#420330
    # https://stats.stackexchange.com/questions/437840/in-machine-learning-how-does-normalization-help-in-convergence-of-gradient-desc/437848#437848
    Xtrain = Xtrain / 255
    Xtest = Xtest / 255

    return Xtrain, ytrain.reshape(1, -1), Xtest, ytest.reshape(1, -1)
