import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Any

from fit import (
    DecayParameters,
    DoubleDecayParameters,
    schmidt,
    double_schmidt,
)


def _visualize_fit(
    data: np.ndarray,
    bins: np.ndarray,
    func: Callable[[Any], float],
    **kwargs,
) -> None:
    x = np.linspace(np.min(bins), np.max(bins), 100)
    yerr = np.sqrt(data)
    plt.errorbar(x=bins, y=data, yerr=yerr, fmt='o')
    
    plt.plot(x, func(x, **kwargs))
    plt.xlabel(r'$ln_{\Delta T}$')
    plt.ylabel('count/channel')
    plt.show()


def visualize_single_fit(
    data: np.ndarray,
    bins: np.ndarray,
    decay: DecayParameters
):
    l,n,c = decay.to_lnc()
    _visualize_fit(
        data=data, bins=bins,
        func=schmidt,
        lamb=l, n=n, c=c
    )


def visualize_double_fit(
    data: np.ndarray,
    bins: np.ndarray,
    decay: DoubleDecayParameters
):
    l1, n1, l2, n2, c = decay.to_lnlnc()
    _visualize_fit(
        data=data, 
        bins=bins,
        func=double_schmidt,
        l1=l1, n1=n1, l2=l2, n2=n2, c=c,
    )
    