# Fast Sampling of Diffusion Models with Exponential Integrator

[Qinsheng Zhang](https://qsh-zh.github.io/), [Yongxin Chen](https://yongxin.ae.gatech.edu/)

A clean implementation for DEIS and iPNDM

# PyTorch Usage

## If score model is trained with discrete time

```python
import torch as th
from th_deis import DisVPSDE, get_sampler
vpsde = DisVPSDE(discrete_alpha) # assume t_start is 0, t_end=len(discrete_alpha) - 1

def eps_fn(x, scalar_t):
    vec_t = (th.ones(x.shape[0])).float().to(x) * scalar_t
    with th.no_grad():
        return eps_model(x, vec_t)
        # ! some model need vec_t shift 1 :(
        # ! check trianing setting of your model 
        # return eps_model(x, vec_t - 1)

sampler_fn = get_sampler(
    vpsde, 
    num_step, 
    eps_fn, 
    order=3, # deis support 0,1,2,3, iPNDM will ignore the arg
    method="deis", # support deis or iPNDM
)

sample = sampler_fn(noise)
```

## Demo: celeba 10 step with an FID less than 7.0 ( 6.26 tested on my Machine)

```shell
cd demo/dis_celeba
bash run.sh
```

## If score model is trained with continuous time

**Not tested yet for the torch! See Jax version for tested usage**

```python
from th_deis import CntVPSDE, get_sampler
vpsde = CntVPSDE(alpha_fn, t_start, t_end)

sampler_fn = get_sampler(
    vpsde, 
    num_step, 
    eps_fn, 
    order=3, # deis support 0,1,2,3, iPNDM will ignore the arg
    method="deis", # support deis or iPNDM
)
```

# Jax Usage


## Reference

```tex
@article{zhang2022fast,
  title={Fast Sampling of Diffusion Models with Exponential Integrator},
  author={Zhang, Qinsheng and Chen, Yongxin},
  journal={arXiv preprint arXiv:2204.13902},
  year={2022}
}
```