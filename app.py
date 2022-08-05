import numpy as np
import matplotlib.pyplot as plt

# variáveis complementares
k = 1
m = 1
t_max = 10
delta = 1

# condições iniciais
x0 = -1
v0 = 1


def a(x): return - (k * x / m)


def V(xt):
    return xt ** 4 - 2 * xt ** 2


def Energy(vt, xt):
    return (1/2) * m * vt ** 2 + V(xt)


def Verlet(t_max, delta_t, x0, v0, a):

    delta_t2 = delta_t * delta_t
    x1 = x0 + v0 * delta_t + 0.5 * a(x0) * delta_t2

    xt_old = x0
    xt = x1

    position = [x0]
    velocity = [v0]
    time = [0]

    for t in np.arange(delta_t, t_max, delta_t):
        x_next = 2 * xt - xt_old + a(xt) * delta_t2
        vt = (x_next - xt_old) / (2 * delta_t)

        position.append(xt)
        time.append(t)
        velocity.append(vt)
        xt_old = xt
        xt = x_next

    return {"x": position, "t": time, "v": velocity}
