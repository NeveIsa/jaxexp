import numpy as np
from jax import grad, jit
import jaxopt
from tqdm import tqdm

@jit
def f(var):
    x,y = var
    return (x-2)**2 + (y-3)**2

init_params = np.array([0.0, 0.0])

solver = jaxopt.GradientDescent(f, maxiter=100, verbose=True)
res = solver.run(init_params)
print("Solution using jaxopt gd:", res.params)

params = init_params.copy()
df = jit(grad(f))
lr=1e-2
iters=1000

pbar = tqdm(range(iters), leave=False)
for i in pbar:
    gradient = df(params)
    params = params - lr*gradient
    pbar.set_postfix({'f': f"{f(params):2f}"})

print("\nSolution using manual gd:", params)
