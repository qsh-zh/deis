import jammy.io as jio
import jax.numpy as jnp
import jax_deis
import numpy as np
import th_deis
import torch as th
from matplotlib import pyplot as plt

jio.makedirs("data")

def test_jax_continuous():
    num_step = 50
    t2alpha_fn, alpha2t_fn = jax_deis.get_linear_alpha_fns(0.01, 20)
    exp_sde = jax_deis.VPSDE(t2alpha_fn, alpha2t_fn, 1e-3, 1.0)
    fig, ax = plt.subplots(1, 1, figsize=(7 * 1, 7 * 1))
    for ts_order, ts_phase in [
        (1, "t"),
        (2, "t"),
        (3, "t"),
        (0, "log"),
        (5, "rho"),
        (6, "rho"),
        (7, "rho"),
        (8, "rho"),
    ]:
        rev_ts = jax_deis.sde.get_rev_ts(exp_sde, num_step, ts_order, ts_phase)
        ax.plot(rev_ts, '-*', label=f"{ts_phase}-{ts_order}")
    fig.legend()
    fig.savefig("data/jax_rev_ts_continuous.png")

def test_jax_discrete():
    num_step = 100
    t2alpha_fn, _ = jax_deis.get_linear_alpha_fns(0.01, 20)
    discrete_alpha = t2alpha_fn(jnp.linspace(0.0, 1.0, 1000+1)) # default ddpm alpha
    exp_sde = jax_deis.vpsde.DiscreteVPSDE(discrete_alpha)
    
    # check rec ts, alpha
    test_ts = jnp.linspace(exp_sde.sampling_eps, exp_sde.sampling_T, num_step)
    test_alpha = exp_sde.t2alpha_fn(test_ts)
    test_rec_ts = exp_sde.alpha2t_fn(test_alpha)
    fig, axs = plt.subplots(1, 2, figsize=(7 * 2, 7 * 1))
    axs[0].plot(test_rec_ts - test_ts)
    axs[0].set_title("reconstruct ts")
    axs[1].plot(exp_sde.t2alpha_fn(test_rec_ts) - test_alpha)
    axs[1].set_title("reconstruct alpha")
    fig.savefig("data/jax_rec_ts_alpha.png")

    fig, ax = plt.subplots(1, 1, figsize=(7 * 1, 7 * 1))
    for ts_order, ts_phase in [
        (1, "t"),
        (2, "t"),
        (3, "t"),
        (0, "log"),
        (5, "rho"),
        (6, "rho"),
        (7, "rho"),
        (8, "rho"),
    ]:
        print(f"{ts_phase}-{ts_order}")
        rev_ts = jax_deis.sde.get_rev_ts(exp_sde, num_step, ts_order, ts_phase)
        ax.plot(rev_ts, '-*', label=f"{ts_phase}-{ts_order}")
    fig.legend()
    fig.savefig("data/jax_rev_ts_discrete.png")


def test_th_continuous():
    num_step = 50
    t2alpha_fn, alpha2t_fn = th_deis.get_linear_alpha_fns(0.01, 20)
    exp_sde = th_deis.VPSDE(t2alpha_fn, alpha2t_fn, 1e-3, 1.0)
    fig, ax = plt.subplots(1, 1, figsize=(7 * 1, 7 * 1))
    for ts_order, ts_phase in [
        (1, "t"),
        (2, "t"),
        (3, "t"),
        (0, "log"),
        (5, "rho"),
        (6, "rho"),
        (7, "rho"),
        (8, "rho"),
    ]:
        rev_ts = th_deis.sde.get_rev_ts(exp_sde, num_step, ts_order, ts_phase)
        ax.plot(rev_ts, '-*', label=f"{ts_phase}-{ts_order}")
    fig.legend()
    fig.savefig("data/th_rev_ts_continuous.png")


def test_th_discrete():
    num_step = 100
    t2alpha_fn, _ = th_deis.get_linear_alpha_fns(0.01, 20)
    discrete_alpha = th.from_numpy(
            np.asarray(t2alpha_fn(jnp.linspace(0.0, 1.0, 1000+1))) # default ddpm alpha
        )
    exp_sde = th_deis.vpsde.DiscreteVPSDE(discrete_alpha)
    
    # check rec ts, alpha
    test_ts = jnp.linspace(exp_sde.sampling_eps, exp_sde.sampling_T, num_step)
    test_alpha = exp_sde.t2alpha_fn(test_ts)
    test_rec_ts = exp_sde.alpha2t_fn(test_alpha)
    fig, axs = plt.subplots(1, 2, figsize=(7 * 2, 7 * 1))
    axs[0].plot(test_rec_ts - test_ts)
    axs[0].set_title("reconstruct ts")
    axs[1].plot(exp_sde.t2alpha_fn(test_rec_ts) - test_alpha)
    axs[1].set_title("reconstruct alpha")
    fig.savefig("data/th_rec_ts_alpha.png")

    fig, ax = plt.subplots(1, 1, figsize=(7 * 1, 7 * 1))
    for ts_order, ts_phase in [
        (1, "t"),
        (2, "t"),
        (3, "t"),
        (0, "log"),
        (5, "rho"),
        (6, "rho"),
        (7, "rho"),
        (8, "rho"),
    ]:
        print(f"{ts_phase}-{ts_order}")
        rev_ts = th_deis.sde.get_rev_ts(exp_sde, num_step, ts_order, ts_phase)
        ax.plot(rev_ts, '-*', label=f"{ts_phase}-{ts_order}")
    fig.legend()
    fig.savefig("data/th_rev_ts_discrete.png")

def test_cos_mapping():
    t2alpha_fn, alpha2t_fn = jax_deis.get_cos_alpha_fns()
    test_ts = jnp.linspace(1e-3, 1.0, 100)
    test_alpha = t2alpha_fn(test_ts)
    test_rec_ts = alpha2t_fn(test_alpha)
    fig, axs = plt.subplots(1, 2, figsize=(7 * 2, 7 * 1))
    axs[0].plot(test_rec_ts - test_ts)
    axs[0].set_title("reconstruct ts")
    axs[1].plot(t2alpha_fn(test_rec_ts) - test_alpha)
    axs[1].set_title("reconstruct alpha")
    fig.savefig("data/cos_ts_alpha.png")
