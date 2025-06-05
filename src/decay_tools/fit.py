import numpy as np
import scipy.optimize as op
import scipy.stats as st
import scipy.integrate
from typing import Literal, Iterable
from dataclasses import dataclass


@dataclass
class DecayParameters:
    hl_us: float
    n0: float
    c: float = 0
    d_hl: float | None = None
    d_n0:           float | None = None
    d_c:            float | None = None

    def to_lnc(self):
        return [float(np.log(2) / self.hl_us), self.n0, self.c]

    def __repr__(self) -> str:
        str_list = []

        name_to_field = {
            "T1/2": [self.hl_us, self.d_hl],
            "n0":   [self.n0, self.d_n0],
            "background constant": [self.c, self.d_c],
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


@dataclass
class DoubleDecayParameters:
    hl_short_us: float
    hl_long_us: float
    n0_short:   float
    n0_long:    float
    c:          float = 0
    d_hl_short: float | None = None
    d_hl_long:  float | None = None
    d_n0_short: float | None = None
    d_n0_long:  float | None = None
    d_c:        float | None = None

    def to_lnlnc(self) -> list[float]:
        l_short = float(np.log(2) / self.hl_short_us)
        l_long = float(np.log(2) / self.hl_long_us)
        return [l_short, self.n0_short, l_long, self.n0_long, self.c]
    
    def get_ratio(self) -> tuple[float, float]:
        n1, n2 = self.n0_short, self.n0_long
        dn1, dn2 = self.d_n0_short, self.d_n0_long

        nu = n2 / (n1 + n2)
        if dn1 is not None and dn2 is not None:
            dnu = (dn1 * dn1 * n2 * n2 + dn2 * dn2 * n1 * n1) ** 0.5 / (n1 + n2) ** 2
        else:
            dnu = None
        return nu, dnu

    def __repr__(self) -> str:
        str_list = []

        name_to_field = {
            "T1/2 short": [self.hl_short_us, self.d_hl_short],
            "T1/2 long": [self.hl_long_us, self.d_hl_long],
            "n0 short":   [self.n0_short, self.d_n0_short],
            "n0 long":   [self.n0_long, self.d_n0_long],
            "long/total ratio": self.get_ratio(),
            "background constant": [self.c, self.d_c],
        }
        
        for name, [v, dv] in name_to_field.items():
            dv = "?" if dv is None else f"{dv:.2f}"
            s = f"{name} = {v:.2f}+-{dv}"
            if name.startswith("T1/2"):
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


def schmidt_integral(logt, lamb, n, c) -> np.ndarray:
    # assume the constant step between logt!
    logt_from = logt - np.diff(logt)[0] / 2
    logt_to = logt + np.diff(logt)[0] / 2
    
    k_from = np.exp(logt_from + np.log(lamb))
    k_to = np.exp(logt_to + np.log(lamb))
    N = n * (np.exp(-k_from) - np.exp(-k_to)) + c * (logt_to - logt_from)
    return N


def double_schmidt_integral(logt, l1, n1, l2, n2, c):
    logt_from = logt - np.diff(logt)[0] / 2
    logt_to = logt + np.diff(logt)[0] / 2
    
    k_from = np.exp(logt_from + np.log(l1))
    k_to = np.exp(logt_to + np.log(l1))
    N1 = n1 * (np.exp(-k_from) - np.exp(-k_to))
    l = l2 / l1
    N2 = n2 * (np.exp(-k_from * l) - np.exp(-k_to * l))
    Nc = c * (logt_to - logt_from)
    N = N1 + N2 + Nc
    return N


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
    n = len(logt) ** (1/3)
    width = 2 * (q75 - q25) / n
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


DATA = np.ndarray
BINS = np.ndarray

def get_hist_and_bins(
    logt: Iterable,
    n_bins: int | None = None,
    n_bins_method: Literal["std", "iqr"] = "iqr",
) -> tuple[DATA, BINS]:
    if not isinstance(logt, np.ndarray):
        logt = np.array(logt)

    if n_bins is None:
        n_bins = estimate_n_bins(logt=logt, method=n_bins_method)
    
    data, bins = np.histogram(logt, bins=n_bins)
    # np.histogram returns borders of each histogram bin 
    # now we need to adjust this to store bin centers
    bins = bins[:-1] + np.diff(bins) / 2
    return data, bins


def fit_single_schmidt(
    data: DATA,
    bins: BINS,
    initial_guess: DecayParameters,
    bounds: tuple[DecayParameters | int, DecayParameters | int] | None = None,
    check_chi_square: bool = True,
    chi_square_nddof: int = 3,
) -> DecayParameters:
    #-------------------------------------
    if bounds is None:
        bounds = (-np.inf, np.inf)
    _bounds = []
    for b in bounds:
        if isinstance(b, DecayParameters):
            # transform half time to exponential decay constant
            b = b.to_lnc()
        elif isinstance(b, (float, int)):
            pass
        else:
            raise ValueError(f"Use 'int' or 'float' or 'DecayParameters' to set a bound! {type(b)} was found!")
        _bounds.append(b)
    
    [l, n, c], pcov = op.curve_fit(
        schmidt_integral, # schmidt,
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
        hl_us=t, d_hl=dt,
        n0=n, d_n0=dn,
        c=c, d_c=dc,
    )

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


def fit_double_schmidt(
    data: DATA,
    bins: BINS,
    initial_guess: DoubleDecayParameters,
    bounds: tuple[DoubleDecayParameters | int, DoubleDecayParameters | int] | None = None,
    check_chi_square: bool = True,
    chi_square_nddof: int = 5,
) -> DecayParameters:
    
    if bounds is None:
        bounds = (-np.inf, np.inf)
    _bounds = []
    for b in bounds:
        if isinstance(b, DoubleDecayParameters):
            # transform half time to exponential decay constant
            b = b.to_lnlnc()
        elif isinstance(b, (float, int)):
            pass
        else:
            raise ValueError(f"Use 'int' or 'float' or 'DecayParameters' to set a bound! {type(b)} was found!")
        _bounds.append(b)

    #---------------------------
    [l1, n1, l2, n2, c], pcov = op.curve_fit(
        f=double_schmidt_integral,# double_schmidt,
        xdata=bins,
        ydata=data,
        p0=initial_guess.to_lnlnc(),
        bounds=_bounds,
        method="trf",
    )

    dl1, dn1, dl2, dn2, dc = np.sqrt(pcov.diagonal())
    if l1 < l2:
        l1, l2 = l2, l1
        dl1, dl2 = dl2, dl1
        n1, n2 = n2, n1
        dn1, dn2 = dn2, dn1
    
    t1 = np.log(2) / l1
    t2 = np.log(2) / l2
    dt1 = np.log(2) * dl1 / l1 / l1
    dt2 = np.log(2) * dl2 / l2 / l2
    res = DoubleDecayParameters(
        hl_short_us=t1, d_hl_short=dt1,
        hl_long_us=t2,  d_hl_long=dt2,
        n0_short=n1,    d_n0_short=dn1,
        n0_long=n2,     d_n0_long=n2,
        c=c,            d_c=dc,
    )
    
    # ------------------------------------
    if check_chi_square:
        try:
            if any(data < 10):
                print("Warning! Some categories have less than 10 counts, chi-square test could be not representative!")
            chi = st.chisquare(data, double_schmidt(bins, l1, n1, l2, n2, c), ddof=chi_square_nddof)
            if chi.pvalue > 0.05:
                print("Good fit!")
            else:
                print("Bad fit!")
            print("chi-square", chi)
        except Exception as e:
            print(f"Chi-square test failed: {e}")
    return res
