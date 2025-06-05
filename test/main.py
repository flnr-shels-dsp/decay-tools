import sys

sys.path.append("src/")

import numpy as np
from decay_tools import (
    DecayParameters,
    DoubleDecayParameters,
    fit_single_schmidt,
    fit_double_schmidt,
    get_hist_and_bins,
    estimate_n_bins,
    visualize_double_fit,
    visualize_single_fit
)


if __name__ == "__main__":
    true_half_life = 8
    true_n0 = 5_000
    times_mks = np.random.exponential(scale=true_half_life / np.log(2), size=true_n0)
    print(f"True T1/2={true_half_life:.2f}; Sampled T1/2={np.mean(times_mks)*np.log(2):.2f}")
    print(f"True n0={true_n0}")
    log_times = np.log(times_mks)
    guess = DecayParameters(true_half_life * 1.2, true_n0*0.7, 0)
    print(estimate_n_bins(log_times, method="std"))
    data, bins = get_hist_and_bins(logt=log_times, n_bins_method="iqr")
    
    res = fit_single_schmidt(data, bins, initial_guess=guess)
    print(res)
    visualize_single_fit(data, bins, res)
    
    true_half_life_2 = 80
    true_n0_2 = 1_000
    times_mks_2 = np.random.exponential(scale=true_half_life_2 / np.log(2), size=true_n0_2)
    print("\n")
    print(f"Second activity: True T1/2={true_half_life_2:.2f}; Sampled T1/2={np.mean(times_mks_2)*np.log(2):.2f}")
    print(f"Second activity: True n0 = {true_n0_2}")
    comb_times_mks = np.append(times_mks, times_mks_2)
    comb_log_times = np.log(comb_times_mks)
    
    data, bins = get_hist_and_bins(logt=comb_log_times, n_bins_method="iqr")
    g = DoubleDecayParameters(
        hl_short_us=0.7 * true_half_life,
        hl_long_us=1.3 * true_half_life_2,
        n0_short=true_n0 * 0.9,
        n0_long=true_n0_2*1.2,
        c=0
    )
    comb_res = fit_double_schmidt(data, bins, g)
    print(comb_res)
    visualize_double_fit(data, bins, comb_res)
