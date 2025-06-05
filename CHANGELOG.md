# Changelog

## [Unreleased]

### Added
- **Integral-based decay functions**  
  - `schmidt_integral(logt, lamb, n, c)`  
    - Computes the integral of the single-component Schmidt decay over each log-time bin.
  - `double_schmidt_integral(logt, l1, n1, l2, n2, c)`  
    - Computes the integral of the two-component Schmidt decay (with constant background) over each log-time bin.

### Changed
- **Curve-fitting routines now use integral forms**  
  In `src/decay_tools/fit.py`:
  - `fit_single_schmidt(...)`  
    - Replaced `schmidt(...)` with `schmidt_integral(...)` in both `curve_fit(...)` and the chi-square test.
  - `fit_double_schmidt(...)`  
    - Replaced `double_schmidt(...)` with `double_schmidt_integral(...)` in both `curve_fit(...)` and the chi-square test.

- **Visualization scaling fix**  
  In `src/decay_tools/visualize.py`:
  ```diff
  - plt.plot(x, func(x, **kwargs))
  + plt.plot(x, func(x, **kwargs) * np.diff(bins)[0])
  ```
  This ensures the fitted curve is correctly scaled to “count per bin” when overlaid on a histogram.


## [v0.1.1] - 2025-05-18

### Fixed
- Bins center adjustment calculation

### Updated
- The example from test directory now uses syntetic data


## [v0.1.0] - 2025-05-06

The very first release of the package

**Features**
- `estimate_n_bins` - to estimate optimal number of bins for given experimental data
- `get_hist_and_bins` - to create histogram, return array of values and bin centers
- `fit_single_schmidt` - to fit single activity decay curve on ln-time space
- `fit_double_schmidt` - to fit double activity decay curve on ln-time space
- `visualize_single_fit` and `visualize_double_fit` - to visualize fit results
