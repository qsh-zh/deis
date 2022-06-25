import jax.numpy  as jnp
import numpy as np
import torch as th

def jax2th(array, th_array=None):
    if th_array is None:
        return th.from_numpy(
            np.asarray(array).copy()
        )
    else:
        return th.from_numpy(
            np.asarray(array).copy()
        ).to(th_array.device)

def th2jax(th_array):
    return jnp.asarray(
        th_array.cpu().numpy()
    )