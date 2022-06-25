import jax.numpy as jnp
import jax_deis
import th_deis
import torch as th

data_shape = (3, 13)

def jax_eps_fn(x, t):
    del t
    return jnp.ones(x.shape)

def th_eps_fn(x, t):
    del t
    return th.ones_like(x)

def test_jax_continuous_vpsde():
    num_step = 10
    t2alpha_fn, alpha2t_fn = jax_deis.get_linear_alpha_fns(0.01, 20)
    exp_sde = jax_deis.VPSDE(t2alpha_fn, alpha2t_fn, 1e-3, 1.0)
    noise = jnp.ones(data_shape)
    for ab_order in [1,2,3]:
        for method in ["t_ab", "rho_ab"]:
            sampler_fn = jax_deis.get_sampler(
                exp_sde,
                jax_eps_fn,
                ts_phase="t",
                ts_order=2.0,
                num_step=num_step,
                method = method,
                ab_order= ab_order, 
            )
            sampler_fn(noise)

    for rk_method in ["1euler", "2heun", "3kutta", "4rk"]:
        sampler_fn = jax_deis.get_sampler(
            exp_sde,
            jax_eps_fn,
            ts_phase="t",
            ts_order=2.0,
            num_step=num_step,
            method = "rho_rk",
            rk_method=rk_method
        )
        sampler_fn(noise)

    sampler_fn = jax_deis.get_sampler(
        exp_sde,
        jax_eps_fn,
        ts_phase="t",
        ts_order=2.0,
        num_step=num_step,
        method = "ipndm",
    )
    sampler_fn(noise)

def test_th_continuous_vpsde():
    num_step = 10
    t2alpha_fn, alpha2t_fn = th_deis.get_linear_alpha_fns(0.01, 20)
    exp_sde = th_deis.VPSDE(t2alpha_fn, alpha2t_fn, 1e-3, 1.0)
    noise = th.ones(data_shape)
    for ab_order in [1,2,3]:
        for method in ["t_ab", "rho_ab"]:
            sampler_fn = th_deis.get_sampler(
                exp_sde,
                th_eps_fn,
                ts_phase="t",
                ts_order=2.0,
                num_step=num_step,
                method = method,
                ab_order= ab_order, 
            )
            sampler_fn(noise)

    for rk_method in ["1euler", "2heun", "3kutta", "4rk"]:
        sampler_fn = th_deis.get_sampler(
            exp_sde,
            th_eps_fn,
            ts_phase="t",
            ts_order=2.0,
            num_step=num_step,
            method = "rho_rk",
            rk_method=rk_method
        )
        sampler_fn(noise)

    sampler_fn = th_deis.get_sampler(
        exp_sde,
        th_eps_fn,
        ts_phase="t",
        ts_order=2.0,
        num_step=num_step,
        method = "ipndm",
    )
    sampler_fn(noise)
