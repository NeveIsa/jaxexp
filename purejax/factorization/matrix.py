from jax import grad,jit
import jax.numpy as jnp
from jax.tree_util import tree_map
import numpy as np
# from matplotlib import pyplot as plt
from tqdm import tqdm
from fire import Fire
import time

@jit
def factorization_loss(factors, target, mask=[]):
    if len(mask)==0:
        mask = np.ones(target.shape)
        
    estimate = factors[0] @ factors[1].T
    diff = mask*(target - estimate)
    return jnp.linalg.norm(diff)

def random_lowrank_matrix(shape=7, rank=1):
    a = np.random.randn(shape, rank)
    b = np.random.randn(shape, rank)   
    return a @ b.T, a, b


def train(iters=1000, lr=0.01):

    np.random.seed(108)
    target_matrix,a,b = random_lowrank_matrix(rank=10)

    # init factors
    factors = np.random.randn(*a.shape), np.random.randn(*b.shape)

    pbar = tqdm(range(iters))

    dloss = grad(factorization_loss)
    dloss = jit(dloss)
    
    for i in pbar:
        if i%10==0:
            loss = factorization_loss(factors, target_matrix)
            pbar.set_postfix({"loss":loss})

        gradient = dloss(factors, target_matrix)
        factors = tree_map(lambda x,y: x - lr*y, factors, gradient)               

        # time.sleep(0.1)

    print (np.linalg.norm(factors[0] - a))
        
if __name__=="__main__":
    Fire(train)

