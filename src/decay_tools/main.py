import numpy as np
from fit import (
    DecayParameters,
    DoubleDecayParameters,
    fit_single_schmidt,
    fit_double_schmidt,
    get_hist_and_bins,
)

from visualize import visualize_single_fit, visualize_double_fit

if __name__ == "__main__":
    times_mks = np.array([2, 2, 2.1, 2.1, 2.7, 2.8, 2.8, 2.8, 2.8, 3, 3.4, 3.5, 3.6, 3.6, 4, 4, 4, 4.1, 4.3, 4.4, 4.6, 4.7, 5, 5.3, 5.3, 5.5, 5.7, 5.9, 6.3, 6.5, 6.7, 6.7, 7, 7.3, 7.4, 7.8, 7.9, 8.2, 8.3, 8.3, 8.5, 9, 10.2, 10.3, 10.4, 10.7, 11.2, 11.2, 11.6, 11.6, 11.7, 12.1, 12.5, 13.1, 13.2, 13.3, 13.6, 13.7, 13.8, 13.8, 16, 16.8, 19.7, 20.3, 21.5, 21.5, 21.6, 22.2, 23.2, 25.5, 26.7, 27.6, 30.6, 31.8, 31.8, 32.1, 34.6, 44.9, 44.9, 47.6, 59, 76.3, 80.8, 95.2, 1323.4, 11220.7, 55098.1, 113904.2])
    log_times = np.log(times_mks)
    guess = DecayParameters(10, 100, 0)
    data, bins = get_hist_and_bins(logt=log_times, n_bins=16)
    
    res = fit_single_schmidt(data, bins, initial_guess=guess)
    print(res)
    print("\n\n")
    
    g = DoubleDecayParameters(hl_short_us=5, hl_long_us=50, n0_short=50, n0_long=20, c=0)
    res = fit_double_schmidt(data, bins, g)
    print(res)
    visualize_double_fit(data, bins, res)


