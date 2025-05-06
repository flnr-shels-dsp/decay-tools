import numpy as np
import scipy.optimize as op
import scipy.stats as st
from typing import Literal, Iterable
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class DecayParameters:
    half_life_us: float
    n0: float
    c: float = 0
    delta_half_life_us: float | None = None
    delta_n0:           float | None = None
    delta_c:            float | None = None

    def to_lnc(self):
        return [float(np.log(2) / self.half_life_us), self.n0, self.c]

    def __repr__(self) -> str:
        str_list = []

        name_to_field = {
            "T1/2": [self.half_life_us, self.delta_half_life_us],
            "n0":   [self.n0, self.delta_n0],
            "background constant": [self.c, self.delta_c]
        }
        
        for name, [v, dv] in name_to_field.items():
            dv = "?" if dv is None else f"{dv:.2f}"
            s = f"{name} = {v:.2f}+-{dv}"
            if name == "T1/2":
                s += " us"
            str_list.append(s)
        return "\n".join(str_list)

    def __str__(self):
        return self.__repr__()


def decay_curve_linear(
    t: Iterable, 
    lam: float,
    n0: float,
) -> np.ndarray:
    return n0 * lam * np.exp(-lam * t)


def _log_curve_zero_bkg(logt: np.ndarray, lamb: float, n: float) -> np.ndarray:
    return n * np.exp(logt + np.log(lamb)) * np.exp(-np.exp(logt + np.log(lamb)))


def schmidt(logt, lamb, n, c) -> np.ndarray:
    """
    # TODO
    """
    return _log_curve_zero_bkg(logt, lamb=lamb, n=n) + c


def double_schmidt(logt, l1, n1, l2, n2, c):
    """
    # TODO
    """
    d1 = _log_curve_zero_bkg(logt, lamb=l1, n=n1)
    d2 = _log_curve_zero_bkg(logt, lamb=l2, n=n2)
    d = d1 + d2 + c
    return d


def _estimate_bin_width_std(logt: np.ndarray) -> float:
    """
    Scott, D. 1979. On optimal and data-based histograms. Biometrika, 66:605-610.
    """
    n = len(logt)
    width = 3.49 * np.std(logt) / (n ** (1 / 3))
    return width

def _estimate_bin_width_iqr(logt: np.ndarray) -> float:
    """
    Izenman, A. J. 1991. Recent developments in nonparametric density estimation.
    Journal of the American Statistical Association, 86(413):205-224.
    """
    q75 = np.quantile(logt, 0.75)
    q25 = np.quantile(logt, 0.25)
    n = len(logt)
    width = 2*(q75 - q25) / (n**(1/3))
    return width


def estimate_n_bins(
    logt: np.ndarray,
    do_round: bool = True,
    method: Literal["std", "iqr"] = "iqr",
):
    """
    # TODO
    """
    if method == "std":
        w = _estimate_bin_width_std(logt)
    elif method == "iqr":
        w = _estimate_bin_width_iqr(logt)
    else:
        raise ValueError(
            "Unknown method for optimal number of bins estimation! "
            f"Expected 'iqr' or 'std' but {method} found."
        )
    n_bins = float((np.max(logt) - np.min(logt)) / w)
    if do_round:
        n_bins = round(n_bins)
    return n_bins


def fit_single_schmidt(
    logt: Iterable,
    initial_guess: DecayParameters,
    bounds: tuple[DecayParameters | int, DecayParameters | int] | None = None,
    n_bins: int | None = None,
    n_bins_method: Literal["std", "iqr"] = "iqr",
    check_chi_square: bool = True,
    chi_square_nddof: int = 3,
) -> DecayParameters:
    if not isinstance(logt, np.ndarray):
        logt = np.array(logt)
    
    if n_bins is None:
        n_bins = estimate_n_bins(logt=logt, method=n_bins_method)
    

    data, bins = np.histogram(logt, bins=n_bins)
    bins = bins[:-1] + np.diff(bins)  # now stores bin centers

    #-------------------------------------
    if bounds is None:
        bounds = (0, DecayParameters(1e-6, 1e6, 1e6))
    _bounds = []
    for b in bounds:
        if isinstance(b, DecayParameters):
            # transform half time to exponential decay constant
            b = b.to_lnc()
        elif isinstance(b, int):
            pass
        else:
            raise ValueError(f"Use 'int' or 'DecayParameters' to set a bound! {type(b)} was found!")
        _bounds.append(b)
    
    [l, n, c], pcov = op.curve_fit(
        schmidt,
        bins,
        data,
        p0=initial_guess.to_lnc(),
        bounds=_bounds,
        method='trf'
    )
    perr = np.sqrt(pcov.diagonal())

    #-------------------------------------
    t = np.log(2) / l
    dt = np.log(2) * perr[0] / l / l
    dn = perr[1]
    dc = perr[2]
    res = DecayParameters(
        half_life_us=t, delta_half_life_us=dt,
        n0=n, delta_n0=dn,
        c=c, delta_c=dc,
    )
    print(f"{l=}")
    print(res)

    #-------------------------------------
    if check_chi_square:
        try:
            if any(data < 10):
                print("Warning! Some categories have less than 10 counts, chi-square test could be not representative!")
            chi = st.chisquare(data, schmidt(bins, l, n, c), ddof=chi_square_nddof)
            if chi.pvalue > 0.05:
                print("Good fit!")
            else:
                print("Bad fit!")
            print("chi-square", chi)
        except Exception as e:
            print(f"Chi-square test failed: {e}")
    return res


def visualize_single_fit(
    logt: Iterable,
    decay_parameters: DecayParameters,
    n_bins: int | None = None,
    n_bins_method: Literal["std", "iqr"] = "iqr",
) -> None:
    if n_bins is None:
        n_bins = estimate_n_bins(logt=logt, method=n_bins_method)
    
    data, bins = np.histogram(logt, bins=n_bins)
    bins = bins[:-1] + np.diff(bins)  # now stores bin centers
    x = np.linspace(bins.min(), bins.max(), 100)
    plt.errorbar(x=bins, y=data, yerr=np.sqrt(data), fmt='o')
    [l,n,c] = decay_parameters.to_lnc()
    plt.plot(x, schmidt(x, l, n, c))
    plt.xlabel(r'$ln_{\Delta T}$')
    plt.ylabel('count/channel')
    plt.show()


# def fit_double_schmidt(df, n_bins, pts_drop, nddof=5):
#     d, b = np.histogram(df.logt, bins=n_bins)
#     dt = np.diff(b)[0]
#     print(f'Bin size: {dt}')
#     b = b + dt / 2

#     bb, dd = b[:-1], d
#     b = b[pts_drop:-1]
#     d = d[pts_drop:]
#     #     print(bb)
#     #     print(dd)
#     [l1, n1, l2, n2, c], pcov = op.curve_fit(double_schmidt, b, d / dt, p0=[0.2, 500, 0.02, 10, 1.0],
#                                              bounds=(0, [1, 3e3, 1, 2e3, 1e6]), method='trf')

#     perr = np.sqrt(pcov.diagonal())
#     x = np.linspace(bb.min(), bb.max(), 100)

#     plt.errorbar(x=bb[:pts_drop], y=dd[:pts_drop], yerr=np.sqrt(dd[:pts_drop]), c='r', fmt='o')
#     plt.errorbar(x=b, y=d, yerr=np.sqrt(d), fmt='o')

#     plt.plot(x, double_schmidt(x, l1, n1, l2, n2, c) * dt);
#     plt.plot(x, schmidt(x, l1, n1, c) * dt)
#     plt.plot(x, schmidt(x, l2, n2, c) * dt)
#     plt.xlabel(r'$ln{\Delta T(ER-SF)}, ln(\mu s)$')
#     plt.ylabel('count/channel')
#     #     plt.savefig('ActivitiesNo250_22.png', dpi=500)

#     t1 = np.log(2) / l1
#     t2 = np.log(2) / l2
#     if t1 <= t2:
#         dt1 = np.log(2) * perr[0] / l1 / l1
#         dt2 = np.log(2) * perr[2] / l2 / l2
#         dn1, dn2 = perr[1], perr[3]
#     else:
#         t1, t2 = t2, t1
#         dt1 = np.log(2) * perr[2] / l2 / l2
#         dt2 = np.log(2) * perr[0] / l1 / l1
#         n1, n2 = n2, n1
#         l1, l2 = l2, l1
#         dn1, dn2 = perr[3], perr[1]

#     nu = n2 / (n1 + n2)
#     dnu = (dn1 * dn1 * n2 * n2 + dn2 * dn2 * n1 * n1) ** 0.5 / (n1 + n2) ** 2

#     func = double_schmidt(b, l1, n1, l2, n2, c) * dt
#     xx = np.linspace(b.min(), b.max(), 100)

#     print(f"l1 = {l1}, l2 = {l2}")
#     print(f"T1/2 short = {t1:.2f}+-{dt1:.2f} us, T1/2 long = {t2:.2f}+-{dt2:.2f} us")
#     print(f'n0 short = {n1:.1f}+-{dn1:.1f}, n0 long = {n2:.1f}+-{dn2:.1f}')
#     print(f"background constant = {c:.3f}+-{perr[-1]:.3f}")
#     print(f"isomer/ground = {nu:.3f}+-{dnu:.3f}")
#     print(f"# events observed = {d.sum()}, integral = {np.trapz(y=double_schmidt(xx, l1, n1, l2, n2, c), x=xx)}")
#     print("chi-square", st.chisquare(d, func, ddof=nddof))



if __name__ == "__main__":
    times_mks = np.array([2, 2, 2.1, 2.1, 2.7, 2.8, 2.8, 2.8, 2.8, 3, 3.4, 3.5, 3.6, 3.6, 4, 4, 4, 4.1, 4.3, 4.4, 4.6, 4.7, 5, 5.3, 5.3, 5.5, 5.7, 5.9, 6.3, 6.5, 6.7, 6.7, 7, 7.3, 7.4, 7.8, 7.9, 8.2, 8.3, 8.3, 8.5, 9, 10.2, 10.3, 10.4, 10.7, 11.2, 11.2, 11.6, 11.6, 11.7, 12.1, 12.5, 13.1, 13.2, 13.3, 13.6, 13.7, 13.8, 13.8, 16, 16.8, 19.7, 20.3, 21.5, 21.5, 21.6, 22.2, 23.2, 25.5, 26.7, 27.6, 30.6, 31.8, 31.8, 32.1, 34.6, 44.9, 44.9, 47.6, 59, 76.3, 80.8, 95.2])
    times_ln = np.log(times_mks)
    guess = DecayParameters(10, 100, 0)
    res = fit_single_schmidt(times_ln, initial_guess=guess)
    visualize_single_fit(logt=times_ln, decay_parameters=res)
