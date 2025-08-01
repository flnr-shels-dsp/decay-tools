import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_and_select_columns(
    filename: str,
    cols: list[str],
) -> pd.DataFrame:
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip()
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input: {missing}")
    return df[cols].copy()


def calculate_group_summary(
    df: pd.DataFrame,
    group_col: str,
    energy_col: str,
    integral_col: str,
    events_col: str,
) -> pd.DataFrame:
    for c in (group_col, energy_col, integral_col, events_col):
        if c not in df.columns:
            raise ValueError(f"Column not found: {c}")

    rows: list[dict] = []
    for group in sorted(df[group_col].unique()):
        grp = df[df[group_col] == group]
        total_int = grp[integral_col].sum()
        total_ev  = grp[events_col].sum()
        if total_int == 0:
            weighted = np.nan
        else:
            weighted = (grp[energy_col] * grp[integral_col]).sum() / total_int
        rows.append({
            "group":           group,
            "total_integral":  total_int,
            "weighted_energy": weighted,
            "event_count":     total_ev,
        })

    return pd.DataFrame(rows)


def sum_integral_and_ion_count(
    df: pd.DataFrame,
    group_col: str,
    integral_col: str,
    ion_charge: float,
    e_charge: float,
    micro_to_coulomb: float = 1e-6,
) -> pd.DataFrame:
    for c in (group_col, integral_col):
        if c not in df.columns:
            raise ValueError(f"Column not found: {c}")

    sums = df.groupby(group_col)[integral_col].sum()
    rows: list[dict] = []
    for group, total_int in sums.items():
        Q    = total_int * micro_to_coulomb
        ions = Q / (ion_charge * e_charge) if Q != 0 else 0.0
        rows.append({
            "group":          group,
            "total_integral": total_int,
            "ion_count":      ions,
        })

    return pd.DataFrame(rows)


def calculate_cross_sections(
    df: pd.DataFrame,
    group_col: str,
    events_col: str,
    integral_col: str,
    t: float,
    eps: float,
    ion_charge: float,
    e_charge: float,
    to_pb: float,
) -> pd.DataFrame:
    for c in (group_col, events_col, integral_col):
        if c not in df.columns:
            raise ValueError(f"Column not found: {c}")

    rows: list[dict] = []
    for group, grp in df.groupby(group_col):
        N         = grp[events_col].sum()
        total_int = grp[integral_col].sum()
        Q         = total_int * 1e-6
        ions      = Q / (ion_charge * e_charge) if Q != 0 else 0.0
        sigma_cm2 = N / (ions * t * eps) if ions != 0 else 0.0
        sigma_pb  = sigma_cm2 * to_pb
        rows.append({
            "group":             group,
            "event_count":       N,
            "ion_count":         ions,
            "cross_section_cm2": sigma_cm2,
            "cross_section_pb":  sigma_pb,
        })

    return pd.DataFrame(rows)


def calculate_cross_section_uncertainty(
    df: pd.DataFrame,
    cross_col: str,
    events_col: str,
) -> pd.DataFrame:
    for c in (cross_col, events_col):
        if c not in df.columns:
            raise ValueError(f"Column not found: {c}")

    out = df.copy()
    σ = out[cross_col].astype(float)
    N = out[events_col].astype(float)

    σ_up  = (N + (1 + np.sqrt(N))) * σ / N
    σ_low = (N - np.sqrt(N))       * σ / N

    out["err_up"]  = σ_up  - σ
    out["err_low"] = σ    - σ_low
    return out


def plot_energy_groups(
    df: pd.DataFrame,
    group_col: str,
    energy_col: str,
    integral_col: str,
) -> None:
    for c in (group_col, energy_col, integral_col):
        if c not in df.columns:
            raise ValueError(f"Column not found: {c}")

    plt.figure(figsize=(10, 6))
    markers = ['o','s','D','^','v','p','*','x']
    for i, grp in enumerate(sorted(df[group_col].unique())):
        sub = df[df[group_col] == grp]
        plt.scatter(
            sub[energy_col],
            sub[integral_col],
            marker=markers[i % len(markers)],
            label=f"group {grp}",
            alpha=0.7,
        )
    plt.xlabel("Median Energy (MeV)")
    plt.ylabel("Total Integral")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_cross_section_vs_energy(
    df: pd.DataFrame,
    energy_col: str,
    cross_col: str,
) -> None:
    for c in (energy_col, cross_col):
        if c not in df.columns:
            raise ValueError(f"Column not found: {c}")

    plt.figure(figsize=(8, 6))
    plt.plot(df[energy_col], df[cross_col], 'o-')
    plt.xlabel("Weighted Energy (MeV)")
    plt.ylabel("Cross Section (pb)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_cross_section_with_errors(
    df: pd.DataFrame,
    energy_col: str,
    cross_col: str,
    err_low_col: str,
    err_up_col: str,
) -> None:
    for c in (energy_col, cross_col, err_low_col, err_up_col):
        if c not in df.columns:
            raise ValueError(f"Column not found: {c}")

    plt.figure(figsize=(8, 6))
    plt.errorbar(
        df[energy_col],
        df[cross_col],
        yerr=[df[err_low_col], df[err_up_col]],
        fmt='o-',
        capsize=5,
    )
    plt.xlabel("Weighted Energy (MeV)")
    plt.ylabel("Cross Section (pb)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_cross_section_with_energy_errors(
    df: pd.DataFrame,
    energy_col: str,
    cross_col: str,
    err_low_col: str,
    err_up_col: str,
    group_col: str,
) -> None:
    for c in (energy_col, cross_col, err_low_col, err_up_col, group_col):
        if c not in df.columns:
            raise ValueError(f"Column not found: {c}")

    ranges = df.groupby(group_col)[energy_col].agg(min='min', max='max')
    x_low = df.apply(lambda r: r[energy_col] - ranges.loc[r[group_col],'min'], axis=1)
    x_up  = df.apply(lambda r: ranges.loc[r[group_col],'max'] - r[energy_col], axis=1)

    plt.figure(figsize=(8, 6))
    plt.errorbar(
        df[energy_col],
        df[cross_col],
        xerr=[x_low, x_up],
        yerr=[df[err_low_col], df[err_up_col]],
        fmt='o-',
        capsize=5,
    )
    plt.xlabel("Weighted Energy (MeV)")
    plt.ylabel("Cross Section (pb)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_shifted_cross_section(
    df_clean: pd.DataFrame,
    summary_df: pd.DataFrame,
    sigma_df: pd.DataFrame,
    substrate_loss: float = 3.9,
    target_loss:    float = 1.08,
) -> tuple:
    sum2 = summary_df.rename(columns={"group":"group","weighted_energy":"energy"})[["group","energy"]]
    sig2 = sigma_df.rename(columns={
        "group":"group",
        "cross_section_pb":"cross",
        "err_low":"err_low",
        "err_up":"err_up"
    })[["group","cross","err_low","err_up"]]
    merged = pd.merge(sum2, sig2, on="group")

    rng = df_clean.groupby("EnergyGroup")["Energy"].agg(min="min", max="max") \
          .reset_index().rename(columns={"EnergyGroup":"group"})
    merged = merged.merge(rng, on="group")

    y           = merged["cross"].to_numpy()
    yerr_lower  = merged["err_low"].tolist()
    yerr_upper  = merged["err_up"].tolist()
    x_center    = merged["energy"]
    xerr_lower  = (x_center - merged["min"]).tolist()
    xerr_upper  = (merged["max"] - x_center).tolist()
    shift       = substrate_loss + target_loss/2
    x_lab       = (x_center - shift).to_numpy()

    plt.figure(figsize=(8,6))
    plt.errorbar(x_lab, y, xerr=[xerr_lower, xerr_upper], yerr=[yerr_lower, yerr_upper],
                 fmt='o-', capsize=5, markersize=6)
    plt.xlabel("Laboratory Energy (MeV)")
    plt.ylabel("Cross Section (pb)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return x_lab, y, xerr_lower, xerr_upper, yerr_lower, yerr_upper


def plot_single_channel(
    data: list[tuple],
    label:    str   = "Channel",
    color:    str   = None,
    marker:   str   = "o",
    title:    str   = "Cross Section",
    xlabel:   str   = "Energy (MeV)",
    ylabel:   str   = "Cross Section (pb)",
) -> None:
    df = pd.DataFrame(data, columns=[
        "Energy", "Value", "Err_Up_Y", "Err_Down_Y", "Err_Down_X", "Err_Up_X"
    ])
    df["Energy"]      /= 1e6
    df["Value"]       *= 1e12
    df["Err_Up_Y"]    *= 1e12
    df["Err_Down_Y"]  *= 1e12
    df["Err_Down_X"]  /= 1e6
    df["Err_Up_X"]    /= 1e6

    plt.figure(figsize=(8,6))
    plt.errorbar(
        df["Energy"], df["Value"],
        xerr=[df["Err_Down_X"], df["Err_Up_X"]],
        yerr=[df["Err_Down_Y"], df["Err_Up_Y"]],
        fmt=marker, capsize=5, label=label,
        color=color, ecolor=color
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_combined_cross_section(
    x_lab,
    y,
    xerr_lower,
    xerr_upper,
    yerr_lower,
    yerr_upper,
    gate_data:    list[tuple] | None = None,
    gate_label:   str               = "Gate Data",
    gate_color:   str               = "gold",
    gate_marker:  str               = "s",
    A1:           float             = 26,
    A2:           float             = 238,
    Q:            float             = -78.2,
) -> None:
    excitation_energy = x_lab * (A2 / (A1 + A2)) + Q

    fig, ax1 = plt.subplots(figsize=(9,6))
    ax1.errorbar(
        x_lab, y,
        xerr=[xerr_lower, xerr_upper],
        yerr=[yerr_lower, yerr_upper],
        fmt='o-', capsize=5, markersize=6,
        label="Your data"
    )

    if gate_data is not None:
        df_gate = pd.DataFrame(
            gate_data,
            columns=["Energy","Value","Err_Up_Y","Err_Down_Y","Err_Down_X","Err_Up_X"]
        )
        df_gate["Energy"]     /= 1e6
        df_gate["Value"]      *= 1e12
        df_gate["Err_Up_Y"]   *= 1e12
        df_gate["Err_Down_Y"] *= 1e12
        df_gate["Err_Down_X"] /= 1e6
        df_gate["Err_Up_X"]   /= 1e6

        ax1.errorbar(
            df_gate["Energy"], df_gate["Value"],
            xerr=[df_gate["Err_Down_X"], df_gate["Err_Up_X"]],
            yerr=[df_gate["Err_Down_Y"], df_gate["Err_Up_Y"]],
            fmt=gate_marker, capsize=5, label=gate_label,
            color=gate_color, ecolor=gate_color
        )

    ax1.set_xlabel("Laboratory Energy (MeV)")
    ax1.set_ylabel("Cross Section (pb)")
    ax1.legend()
    ax1.grid(True)

    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(x_lab)
    ax2.set_xticklabels([f"{val:.1f}" for val in excitation_energy])
    ax2.set_xlabel("Excitation Energy (MeV)")

    plt.tight_layout()
    plt.show()


def build_cross_section_table(
    df_clean:  pd.DataFrame,
    summary_df: pd.DataFrame,
    sigma_df:   pd.DataFrame,
) -> pd.DataFrame:
    sum2 = (
        summary_df
        .rename(columns={"group":"group","weighted_energy":"energy"})
        [["group","energy"]]
    )
    sig2 = (
        sigma_df
        .rename(columns={
            "group":"group",
            "cross_section_pb":"cross",
            "err_low":"err_low",
            "err_up":"err_up"
        })
        [["group","cross","err_low","err_up"]]
    )
    rng = (
        df_clean
        .groupby("EnergyGroup")["Energy"]
        .agg(min="min", max="max")
        .reset_index()
        .rename(columns={"EnergyGroup":"group"})
    )
    rng = rng.merge(sum2, on="group")
    rng["err_energy_low"] = rng["energy"] - rng["min"]
    rng["err_energy_up"]  = rng["max"]    - rng["energy"]

    merged = (
        sum2
        .merge(sig2, on="group")
        .merge(rng[["group","err_energy_low","err_energy_up"]], on="group")
    )

    return pd.DataFrame({
        "group":          merged["group"],
        "energy":         merged["energy"],
        "err_energy_low": merged["err_energy_low"],
        "err_energy_up":  merged["err_energy_up"],
        "cross":          merged["cross"],
        "err_low":        merged["err_low"],
        "err_up":         merged["err_up"],
    })


def build_group_summary(
    df: pd.DataFrame,
    group_col: str,
    energy_col: str,
) -> pd.DataFrame:
    if group_col not in df.columns or energy_col not in df.columns:
        raise ValueError(f"Column not found: {group_col} or {energy_col}")

    agg = df.groupby(group_col)[energy_col].agg(min="min", max="max", mean="mean")
    agg["err_energy_low"] = agg["mean"] - agg["min"]
    agg["err_energy_up"]  = agg["max"]  - agg["mean"]

    agg = agg.reset_index().rename(columns={
        group_col:   "group",
        "mean":      "energy_mean"
    })

    return agg[["group", "energy_mean", "err_energy_low", "err_energy_up"]]
