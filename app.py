from cmath import sqrt
import numpy as np
import matplotlib.pyplot as plt

# variáveis complementares
k = 1
m = 1
t_max = 10
delta = 0.1

# condições iniciais
x0 = -1
v0 = 1


def a(x): return - (4 * x ** 3 - 4 * x) / m


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


def RK2(t_max, delta, x0, v0, a):
    position = [x0]
    velocity = [v0]
    time = [0]

    xt = x0
    vt = v0

    for t in np.arange(delta, t_max, delta):
        x2 = xt + vt * delta / 2
        v2 = vt + a(xt) * delta / 2

        xt += v2 * delta
        vt += a(x2) * delta

        position.append(xt)
        time.append(t)
        velocity.append(vt)

    return {"x": position, "t": time, "v": velocity}


def RK4(delta, final_t, x0, v0, a):
    position = [x0]
    velocity = [v0]
    time = [0]

    v = v0
    x = x0

    for t in np.arange(delta, final_t, delta):
        a1 = a(x)
        v1 = v

        x2 = x + v1 * delta / 2
        v2 = v + a1 * delta / 2
        a2 = a(x2)

        x3 = x + v2 * delta / 2
        v3 = v + a2 * delta / 2
        a3 = a(x3)

        x4 = x + v3 * delta
        v4 = v + a3 * delta
        a4 = a(x4)

        x += (v1 + 2*v2 + 2*v3 + v4) * delta / 6
        v += (a1 + 2*a2 + 2*a3 + a4) * delta / 6

        time.append(t)
        position.append(x)
        velocity.append(v)

    return {"x": position, "t": time, "v": velocity}


def error(v, x):
    E0 = Energy(v[0], x[0])
    soma = 0

    for i in range(0, len(x)):
        Et = Energy(v[i], x[i])

        soma += (Et - E0) ** 2

    err = sqrt(soma / len(x))
    return err

# Errors


verletErr = []
rk2Err = []
rk4Err = []
deltas = []


for i in range(18):
    verlet = Verlet(t_max, delta, x0, v0, a)
    rk2 = RK2(t_max, delta, x0, v0, a)
    rk4 = RK4(delta, t_max, x0, v0, a)

    verletErr.append(error(verlet['v'], verlet['x']))
    rk2Err.append(error(rk2['v'], rk2['x']))
    rk4Err.append(error(rk4['v'], rk4['x']))
    deltas.append(delta)

    delta = delta / 2

partVereltErr = verletErr[:9]
partRK2Err = rk2Err[:9]
partRK4Err = rk4Err[:9]
partDeltas = deltas[:9]


# Erro em escala normal

plt.plot(partDeltas, partVereltErr, label="verlet", marker=".")
plt.plot(partDeltas, partRK2Err, label="RK2",  marker=".")
plt.plot(partDeltas, partRK4Err, label="RK4",  marker=".")
plt.title("Escala normal")
plt.xlabel("dt")
plt.ylabel("erro")
plt.legend()
plt.savefig("normal")
plt.show()


# # Erro em escala semi-log x

plt.semilogx(partDeltas, partVereltErr, label="verlet", marker=".")
plt.semilogx(partDeltas, partRK2Err, label="RK2",  marker=".")
plt.semilogx(partDeltas, partRK4Err, label="RK4",  marker=".")
plt.title("Escala semi-log x")
plt.xlabel("dt")
plt.ylabel("erro")
plt.legend()
plt.savefig("logx")
plt.show()


# Erro em escala semi-log y

plt.semilogy(partDeltas, partVereltErr, label="verlet", marker=".")
plt.semilogy(partDeltas, partRK2Err, label="RK2",  marker=".")
plt.semilogy(partDeltas, partRK4Err, label="RK4",  marker=".")
plt.title("Escala semi-log y")
plt.xlabel("dt")
plt.ylabel("erro")
plt.legend()
plt.savefig("logy")
plt.show()

# Erro em escala log-log

plt.loglog(deltas, verletErr, label="verlet", marker=".")
plt.loglog(deltas, rk2Err, label="RK2",  marker=".")
plt.loglog(deltas, rk4Err, label="RK4",  marker=".")
plt.title("Escala log-log")
plt.xlabel("dt")
plt.ylabel("erro")
plt.legend()
plt.savefig("loglog")
plt.show()
