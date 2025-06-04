import sys

sys.path.append("src/")

import numpy as np
from decay_tools.fit import (
    schmidt
)
import matplotlib.pyplot as plt
from scipy import integrate


def integ(t1, t2, lamb, n0):
    k1 = np.exp(t1 + np.log(lamb))
    k2 = np.exp(t2 + np.log(lamb))
    return n0*(np.exp(-k1) - np.exp(-k2))


if __name__ == "__main__":
    true_half_life = 8
    n0 = 5_000
    lt = np.linspace(np.log(1e-1), np.log(100), 500)
    lamb = np.log(2) / true_half_life
    f = schmidt(lt, lamb, n=n0, c=0)
    t1 = np.log(10)
    t2 = np.log(20)

    i1 = integ(t1, t2, lamb, n0)
    print(i1)
    x = np.linspace(t1, t2, 500)
    i2 = integrate.trapezoid(
        y=schmidt(x, lamb, n=n0, c=0),
        x=x,
    )
    print(i2)
    print(i2 / i1)
    print(np.diff(x)[0])
