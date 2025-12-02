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
