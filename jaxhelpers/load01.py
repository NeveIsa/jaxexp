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
    Xtrain = Xtrain / 255
    Xtest = Xtest / 255

    return Xtrain, ytrain.reshape(1, -1), Xtest, ytest.reshape(1, -1)
