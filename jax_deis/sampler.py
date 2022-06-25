import jax
import jax.numpy as jnp

from .multistep import ab_step, get_ab_eps_coef
from .rk import get_rk_fn
from .sde import MultiStepSDE, get_rev_ts
from .vpsde import VPSDE

def get_sampler(sde, eps_fn, ts_phase, ts_order, num_step, method="rho_rk",ab_order=3, rk_method="3kutta"):
    if method.lower() == "rho_rk":
        return get_sampler_rho_rk(sde, eps_fn, ts_phase, ts_order, num_step, rk_method)
    elif method.lower() == "rho_ab":
        return get_sampler_rho_ab(sde, eps_fn, ts_phase, ts_order, num_step, ab_order)
    elif method.lower() == "t_ab":
        return get_sampler_t_ab(sde, eps_fn, ts_phase, ts_order, num_step, ab_order)
    elif method.lower() == "ipndm":
        return get_sampler_ipndm(sde, eps_fn, num_step)
    raise RuntimeError(f"{method} not support!!")


def get_sampler_t_ab(sde, eps_fn, ts_phase, ts_order, num_step, ab_order):
    rev_ts = get_rev_ts(sde, num_step, ts_order, ts_phase=ts_phase)
    
    x_coef = sde.psi(rev_ts[:-1], rev_ts[1:])
    eps_coef = get_ab_eps_coef(sde, ab_order, rev_ts, ab_order)
    ab_coef = jnp.concatenate([x_coef[:, None], eps_coef], axis=1)

    def sampler(x0):
        def ab_body_fn(i, val):
            x, eps_pred = val
            s_t= rev_ts[i]
            
            new_eps = eps_fn(x, s_t)
            new_x, new_eps_pred = ab_step(x, ab_coef[i], new_eps, eps_pred)
            return new_x, new_eps_pred


        eps_pred = jnp.asarray([x0,] * ab_order)
        img, _ = jax.lax.fori_loop(0, num_step, ab_body_fn, (x0, eps_pred))
        return img
    return sampler

def get_sampler_ipndm(sde, eps_fn, num_step):
    assert isinstance(sde, VPSDE)
    rev_ts = get_rev_ts(sde, num_step, 1, ts_phase="t")
    
    x_coef = sde.psi(rev_ts[:-1], rev_ts[1:]) #(n, )

    def get_linear_ab_coef(i):
        if i == 0:
            return jnp.asarray([1.0, 0, 0, 0]).reshape(-1,4)
        prev_coef = get_linear_ab_coef(i-1)
        cur_coef = None
        if i == 1:
            cur_coef = jnp.asarray([1.5, -0.5, 0, 0])
        elif i == 2:
            cur_coef = jnp.asarray([23, -16, 5, 0]) / 12.0
        else:
            cur_coef = jnp.asarray([55, -59, 37, -9]) / 24.0
        return jnp.concatenate(
            [prev_coef, cur_coef.reshape(-1,4)]
        )
    linear_ab_coef = get_linear_ab_coef(len(rev_ts) - 2) # (n, 4)

    next_ts, cur_ts = rev_ts[1:], rev_ts[:-1]
    next_alpha, cur_alpha = sde.t2alpha_fn(next_ts), sde.t2alpha_fn(cur_ts)
    ddim_coef = jnp.sqrt(1 - next_alpha) - jnp.sqrt(next_alpha / cur_alpha) * jnp.sqrt(1 - cur_alpha) # (n,)

    eps_coef = ddim_coef.reshape(-1,1) * linear_ab_coef

    ei_ab_coef = jnp.concatenate([x_coef[:, None], eps_coef], axis=1)

    def sampler(x0):
        def ab_body_fn(i, val):
            x, eps_pred = val
            s_t= rev_ts[i]
            
            new_eps = eps_fn(x, s_t)
            new_x, new_eps_pred = ab_step(x, ei_ab_coef[i], new_eps, eps_pred)
            return new_x, new_eps_pred


        eps_pred = jnp.asarray([x0,] * 3)
        x_eps, _ = jax.lax.fori_loop(0, num_step, ab_body_fn, (x0, eps_pred))
        return x_eps
    return sampler


def get_sampler_rho_ab(sde, eps_fn, ts_phase, ts_order, num_step, ab_order):
    rev_ts = get_rev_ts(sde, num_step, ts_order, ts_phase=ts_phase)
    
    highest_order = ab_order
    x_coef = jnp.ones(rev_ts.shape[0]-1)
    rev_rhos = sde.t2rho(rev_ts)
    class HelperSDE(MultiStepSDE):
        def psi(cls, t1, t2):
            return t1 / t1 * t2 / t2
        def eps_integrand(cls, vec_t):
            return vec_t / vec_t
    eps_ab_coef = get_ab_eps_coef(HelperSDE(), highest_order, rev_rhos, ab_order)
    ab_coef = jnp.concatenate([x_coef[:, None], eps_ab_coef], axis=1)
    nfe = len(rev_ts) - 1

    @jax.jit
    def eps_fn_vrho(v, rho):
        t = sde.rho2t(rho)
        x = sde.v2x(v, t)
        return eps_fn(x, t)
    
    def sampler(xT):
        vT = sde.x2v(xT, rev_ts[0])
        def ab_body_fn(i, val):
            v_cur, eps_prev_preds = val # eps_preds (highest_order, ) start with prev_rho
            rho_cur = rev_rhos[i]
            eps_cur = eps_fn_vrho(v_cur, rho_cur)

            v_next, new_eps_cur_preds = ab_step(v_cur, ab_coef[i], eps_cur, eps_prev_preds)
            return v_next, new_eps_cur_preds

        eps_pred = jnp.asarray([vT,] * highest_order)
        v_eps, _ = jax.lax.fori_loop(0, nfe, ab_body_fn, (vT, eps_pred))
        x_eps = sde.v2x(v_eps, rev_ts[-1])
        return x_eps
    return sampler


def get_sampler_rho_rk(sde, eps_fn, ts_phase, ts_order, num_step, rk_method):
    rev_ts = get_rev_ts(sde, num_step, ts_order, ts_phase=ts_phase)
    rk_fn = get_rk_fn(rk_method)
    rev_rhos = sde.t2rho(rev_ts)

    @jax.jit
    def eps_fn_vrho(v, rho):
        t = sde.rho2t(rho)
        x = sde.v2x(v, t)
        return eps_fn(x, t)

    def _step_fn(i_th, v):
        rho_cur, rho_next = rev_rhos[i_th], rev_rhos[i_th + 1]
        delta_t = rho_next - rho_cur
        return rk_fn(v, rho_cur, delta_t, eps_fn_vrho)

    def sample_fn(xT):
        vT = sde.x2v(xT, rev_ts[0])
        veps = jax.lax.fori_loop(0, len(rev_rhos)-1, _step_fn, vT)
        xeps = sde.v2x(veps, rev_ts[-1])
        return xeps

    return sample_fn
