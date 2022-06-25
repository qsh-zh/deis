# Fast Sampling of Diffusion Models with Exponential Integrator

[Qinsheng Zhang](https://qsh-zh.github.io/), [Yongxin Chen](https://yongxin.ae.gatech.edu/)

A clean implementation for DEIS and iPNDM

![deis](assets/fig1.png)


# Update

* **BREAKING CHANGE**: [v1.0](https://github.com/qsh-zh/deis/tree/v1.0) API changes greatly as we add `ρRK-DEIS` and `ρAB-DEIS` algorithms and more choice for time scheduling. If you are only interested in `tAB-DEIS` / `iPNDM` or previous codebase, check [v0.1](https://github.com/qsh-zh/deis/tree/v0.1)

# Usage

```shell
# for pytorch user
pip install "jax[cpu]"
```

## If diffusion models are trained with continuous time

```py
import jax_deis as deis

def eps_fn(x_t, scalar_t):
    vec_t = jnp.ones(x_t.shape[0]) * scalar_t
    return eps_model(x_t, vec_t)

# pytorch
# import th_deis as deis
# def eps_fn(x_t, scalar_t):
#     vec_t = (th.ones(x_t.shape[0])).float().to(x_t) * scalar_t
#     with th.no_grad():
#         return eps_model(x_t, vec_t)

# mappings between t and alpha in VPSDE
# we provide popular linear and cos mappings
t2alpha_fn,alpha2t_fn = deis.get_linear_alpha_fns(beta_0=0.01, beta_1=20)

vpsde = deis.VPSDE(
    t2alpha_fn, 
    alpha2t_fn,
    sampling_eps, # sampling end time t_0
    sampling_T # sampling starting time t_T
)

sampler_fn = deis.get_sampler(
    # args for diffusion model
    vpsde,
    eps_fn,
    # args for timestamps scheduling
    ts_phase="t", # support "rho", "t", "log"
    ts_order=2.0,
    num_step=10,
    # deis choice
    method = "t_ab", # deis sampling algorithms: support "rho_rk", "rho_ab", "t_ab", "ipndm"
    ab_order= 3, # greater than 0, used for "rho_ab", "t_ab" algorithms, other algorithms will ignore the arg
    rk_method="3kutta" # used for "rho_rk" algorithms, other algorithms will ignore the arg
)

sample = sampler_fn(noise)
```

## If diffusion models are trained with discrete time

```py
#! by default the example assumes sampling 
#! from t=len(discrete_alpha) - 1 to t=0
#! totaly len(discrete_alpha) steps if we use delta_t = 1
vpsde = deis.DiscreteVPSDE(discrete_alpha)
```

# A short derivation for DEIS

<details>
<summary>Exponential integrator in diffusion model</summary>

The key insight of exponential integrator is taking advantages of all math structure present in ODEs. The goal is to reduce discretization error as small as possible. 

The math structure presents in diffusion models, including semilinear structure, analystic formula for drift and diffusion coefficients.

Below we present a short derivation for applications of exponential integrator in diffusion model.

## Forward SDE

$$
dx = F_tx dt + G_td\mathbf{w}
$$

## Backward ODE

$$
dx = F_tx dt + 0.5 G_tG_t^T L_t^{-T} \epsilon(x, t) dt
$$

where $L_t L_t^{T} = \Sigma_t$ and $\Sigma_t$ are variance of $p_{0t}(x_t | x_0)$.

## Exponential Integrator

We can get rid of semilinear structure with **Exponential Integrator** by introducing a new variable $y$

$$
y_t = \Psi(t) x_t \quad \Psi(t) = \exp{-\int_0^{t} F_\tau d \tau}
$$

And ODE is simplified into

$$
\dot{y}_t = 0.5 \Psi(t) G_t G_t^T L_t^{-T} \epsilon(x(y), t)
$$

where $x(y)$ maps $y_t$ to $x_t$.


## Time scaling

We can take one step further when $F_t, G_t$ are scalars by rescaling time

$$
\dot{v}_\rho = \epsilon(x(v), t(\rho))
$$

where $y_t = v_\rho$ and $d \rho = 0.5 \Psi(t) G_t G_t^T L_t^{-T} dt$. And $x(v)$ maps $v_\rho$ to $x_t$ and $t(\rho)$ maps $\rho$ to $t$.

## High order solver

By absorbing all math structure, we reach the following ODE

$$
\dot{v}_\rho = \epsilon(x(v), t(\rho))
$$

Then we can use well-established ODE solver, such as multistep and runge kutta.
</details>

# Demo

- [continuous vpsde](demo/cnt_cifar/deis.ipynb) Based on [score_sde codebase](https://github.com/yang-song/score_sde)
- [discrete vpsde](demo/discrete_celeba) Based on [PNDM codebase](https://github.com/luping-liu/PNDM)


# Reference

```tex
@article{zhang2022fast,
  title={Fast Sampling of Diffusion Models with Exponential Integrator},
  author={Zhang, Qinsheng and Chen, Yongxin},
  journal={arXiv preprint arXiv:2204.13902},
  year={2022}
}
```