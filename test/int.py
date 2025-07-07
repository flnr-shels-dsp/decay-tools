import sys

sys.path.append("src/")

import numpy as np
from decay_tools.half_live.fit import (
    schmidt,
    double_schmidt_integral,
    double_schmidt
)
import matplotlib.pyplot as plt
from scipy import integrate


def integ(t1, t2, l1, n1, l2, n2, c):
    k_from = np.exp(t1 + np.log(l1))
    k_to = np.exp(t2 + np.log(l1))
    N1 = n1 * (np.exp(-k_from) - np.exp(-k_to))
    l = l2 / l1
    N2 = n2 * (np.exp(-k_from * l) - np.exp(-k_to * l))
    Nc = c * (t2 - t1)
    N = N1 + N2 + Nc
    return N


if __name__ == "__main__":
    true_half_life = 8
    n0 = 5_000
    true_half_life_2 = 50
    n0_2 = 1000

    lamb = np.log(2) / true_half_life
    lamb_2 = np.log(2) / true_half_life_2
    
    # f = double_schmidt(lt, l1=lamb, l2=lamb_2, n1=n0, n2=n0_2, c=0)
    t1 = np.log(10)
    t2 = np.log(20)
    lt = np.linspace(t1, t2, 500)

    i1 = integ(t1, t2, l1=lamb, l2=lamb_2, n1=n0, n2=n0_2, c=3)
    print(i1)
    x = np.linspace(t1, t2, 500)
    i2 = integrate.trapezoid(
        y=double_schmidt(lt, l1=lamb, l2=lamb_2, n1=n0, n2=n0_2, c=3),
        x=x,
    )
    print(i2)
