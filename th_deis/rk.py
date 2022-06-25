# https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods

def rk4(x, t, dt, fn):
    grad_1 = fn(x, t)
    x_2 = x + grad_1 * dt / 2
    
    grad_2 = fn(x_2, t + dt / 2)
    x_3 = x + grad_2 * dt / 2
    
    grad_3 = fn(x_3, t + dt / 2)
    x_4 = x + grad_3 * dt
    
    grad_4 = fn(x_4, t + dt)
    return x + dt / 6 * (grad_1 + 2 * grad_2 + 2 * grad_3 + grad_4)

def rk3_kutta(x, t, dt, fn):
    c2, c3 = 0.5, 1.0
    a21, a31, a32 = 0.5, -1.0, 2.0
    b1, b2, b3 = 1.0 / 6, 4.0 /6, 1.0 /6

    k1 = fn(x, t)
    k2 = fn(x + dt * a21 * k1, t + dt * c2 )
    k3 = fn(x + dt * a31 * k1 + dt * a32 * k2, t + dt * c3)
    return x + dt * (b1 * k1 + b2 * k2 + b3 * k3)

def rk3_ral(x, t, dt, fn):
    c2, c3 = 1.0 / 2, 3.0 / 4
    a21, a31, a32 = 1.0 / 2, 0.0, 3.0/4
    b1, b2, b3 = 2.0 / 9, 1.0 /3, 4.0 /9
    k1 = fn(x, t)
    k2 = fn(x + dt * a21 * k1, t + dt * c2 )
    k3 = fn(x + dt * a31 * k1 + dt * a32 * k2, t + dt * c3)
    return x + dt * (b1 * k1 + b2 * k2 + b3 * k3)

def rk3_heun(x, t, dt, fn):
    c2, c3 = 1.0 / 3, 2.0 / 3
    a21, a31, a32 = 1.0 / 3, 0.0, 2.0/3
    b1, b2, b3 = 1.0 / 4, 0.0, 3.0 /4
    k1 = fn(x, t)
    k2 = fn(x + dt * a21 * k1, t + dt * c2 )
    k3 = fn(x + dt * a31 * k1 + dt * a32 * k2, t + dt * c3)
    return x + dt * (b1 * k1 + b2 * k2 + b3 * k3)

def rk3_vdh(x, t, dt, fn):
    c2, c3 = 8.0/15, 2.0 / 3
    a21, a31, a32 = 8.0/15, 1.0 / 4, 5.0/12
    b1, b2, b3 = 1.0 / 4, 0.0, 3.0 /4
    k1 = fn(x, t)
    k2 = fn(x + dt * a21 * k1, t + dt * c2 )
    k3 = fn(x + dt * a31 * k1 + dt * a32 * k2, t + dt * c3)
    return x + dt * (b1 * k1 + b2 * k2 + b3 * k3)

def rk3_ssprk(x, t, dt, fn):
    c2, c3 = 1.0, 0.5
    a21, a31, a32 = 1.0, 0.25, 0.25
    b1, b2, b3 = 1.0 / 6, 1.0/6, 2.0 /3
    k1 = fn(x, t)
    k2 = fn(x + dt * a21 * k1, t + dt * c2 )
    k3 = fn(x + dt * a31 * k1 + dt * a32 * k2, t + dt * c3)
    return x + dt * (b1 * k1 + b2 * k2 + b3 * k3)

def rk2_heun(x, t, dt, fn):
    grad_1 = fn(x, t)
    grad_2 = fn(x + grad_1 * dt, t + dt)

    return x + dt / 2 * (grad_1 + grad_2)

def rk1_euler(x, t, dt, fn):
    grad_1 = fn(x, t)
    return x + dt * grad_1

def get_rk_fn(name):
    rk_fns = {
        "1euler":rk1_euler,
        "2heun":rk2_heun,
        "3kutta":rk3_kutta,
        "3ral":rk3_ral,
        "3heun":rk3_heun,
        "3vdh":rk3_vdh,
        "3ssprk":rk3_ssprk,
        "4rk":rk4,
    }
    if name in rk_fns:
        return rk_fns[name]
    methods = "\n\t".join(rk_fns.keys())
    raise RuntimeError("Only support rk method\n" + methods)