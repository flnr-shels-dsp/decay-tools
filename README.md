# decay-tools
[![PyPI version](https://img.shields.io/pypi/v/decay-tools)](https://pypi.org/project/decay-tools/)  [![License](https://img.shields.io/github/license/flnr-shels-dsp/decay-tools)](https://github.com/flnr-shels-dsp/decay-tools/blob/main/LICENSE)

Tools for analyzing decay time distributions in nuclear physics experiments


# Installation

```
pip install decay-tools
```

# Usage

## Prepare data

```python
import numpy as np
from decay_tools import estimate_n_bins, get_hist_and_bins

times_mks = np.array([2, 2, ..., 55098.1, 113904.2])
# convert to ln(Δt)
logt = np.log(times_mks)

# find an “optimal” number of bins (IQR or STD method)
nbins = estimate_n_bins(logt, method="iqr")
print(f"Suggested bins: {nbins}")

# build histogram & get bin centers
data, bins = get_hist_and_bins(logt=logt) # default: n_bins_method="iqr"
# or explicitly:
# data, bins = get_hist_and_bins(logt=logt, n_bins=10)
# data, bins = get_hist_and_bins(logt=logt, n_bins_method="std")
```

## Fit a single-component decay curve

```python
from decay_tools import DecayParameters, fit_single_schmidt, visualize_single_fit

# initial guess: half-life=10 μs, N₀=100, background=0
guess = DecayParameters(half_life_us=10, n0=100, c=0)

result = fit_single_schmidt(
    data,
    bins,
    initial_guess=guess,
    check_chi_square=True, # will print chi-squre test results
)

print(result)
# if you want to see the fit overlayed on the histogram:
visualize_single_fit(data, bins, result)
```

## Fit a double-component decay curve

```python
from decay_tools import DoubleDecayParameters, fit_double_schmidt, visualize_double_fit

# initial guess for short & long components
g = DoubleDecayParameters(
    hl_short_us=5, 
    hl_long_us=50,
    n0_short=50, 
    n0_long=20, 
    c=0
)

result = fit_double_schmidt(
    data, 
    bins, 
    initial_guess=g,
)

print(result)
visualize_double_fit(data, bins, result)
```

## Set boundaries

Both `fit_single_schmidt` and `fit_double_schmidt` accept an optional bounds argument to constrain the fit parameters. Pass a tuple `(lower, upper)`, where each bound can be either a `DecayParameters` (or `DoubleDecayParameters`) instance—whose fields are automatically converted to the log-domain constants—or a scalar/int, which applies the same limit to all parameters.

**Note**: The fiting procedure is done over decay constant but not half-life. So, when we set a bound of half-life we actually set the lower bound. If the initial guess for half live is lower than bound, the exception will be raised.

```python
from decay_tools import DecayParameters, fit_single_schmidt

bounds = (
    0,
    DecayParameters(half_life_us=10, n0=1e3,  c=0.1)  # NOTE: we actually limit here the lower bound for half life
)
result = fit_single_schmidt(data, bins, initial_guess=guess, bounds=bounds)
```


## Statistical tests to check if there more than one components in the decay curve

**TBD**

## Cross-sections module

At the input must be provided a dataframe with the format:

```
FileName,MedianEnergy,TotalIntegral,CorrelationsCount
file1,134.2,14890,1
file2,133.3,22540,5
...
```

Then we could categorize the data based on the provided cluster count::

```
df_clean, stats = cluster_and_categorize_energy(
    df=df,
    energy_column_name="MedianEnergy",
    n_clusters=5
)
```

The visualization of the clustered data can be shown with:

```
plot_energy_groups(
    df=df_clean,
    group_col="EnergyGroup",
    energy_col="MedianEnergy",
    integral_col="TotalIntegral",
)
```

Display the beam dose per cluster:

```
ion_df = sum_integral_and_ion_count(
    df_clean,
    "EnergyGroup",
    "TotalIntegral",
    10.0,
)
display(ion_df)
```

Calculation of cross-sections with uncertainties:

```
sigma_df = calculate_cross_sections(
    df_clean,
    group_col="EnergyGroup",
    events_col="Total",
    integral_col="Total Integral",
    t=7.26e17,     
    eps=0.02,            
    ion_charge=10.0,         
)

sigma_df = calculate_cross_section_uncertainty(
    sigma_df,
    cross_col="cross_section_pb",
    events_col="event_count"
)
```

Display the result:

```
plot_cross_section_with_errors(
    merged,
    energy_col="energy",
    cross_col="cross",
    err_low_col="err_low",
    err_up_col="err_up"
)
```