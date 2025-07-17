import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt

def read_and_select_columns(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip()
    selected_columns = ["File", "Median Energy", "Total Integral", "Total"]
    df_selected = df[selected_columns]
    return df_selected

def cluster_by_energy(df: pd.DataFrame, n_groups: int = 5) -> pd.DataFrame:
    df.columns = df.columns.str.strip()

    energies = df["Median Energy"].dropna().values.reshape(-1, 1)

    kmeans = KMeans(n_clusters=n_groups, random_state=0)
    labels = kmeans.fit_predict(energies)

    df_clean = df[df["Median Energy"].notna()].copy()
    df_clean["GroupLabel"] = labels
    df_clean["Energy"] = df_clean["Median Energy"]

    cluster_centers = kmeans.cluster_centers_.flatten()
    sorted_centers = np.sort(cluster_centers)

    boundaries = []
    for i in range(len(sorted_centers) - 1):
        midpoint = (sorted_centers[i] + sorted_centers[i+1]) / 2
        boundaries.append(midpoint)

    min_energy = df_clean["Energy"].min()
    max_energy = df_clean["Energy"].max()
    boundaries = [min_energy - 1e-6] + boundaries + [max_energy + 1e-6]

    def assign_group(energy):
        for i in range(n_groups):
            if boundaries[i] < energy <= boundaries[i+1]:
                return i
        return -1

    df_clean["EnergyGroup"] = df_clean["Energy"].apply(assign_group)

    for i in range(n_groups):
        group = df_clean[df_clean["EnergyGroup"] == i]
        if group.empty:
            continue
        group_min = round(group["Energy"].min(), 2)
        group_max = round(group["Energy"].max(), 2)
        group_center = round(group["Energy"].mean(), 2)

        print(f"\n=== Группа {i+1} ===")
        print(f"Центральная энергия: {group_center} МэВ")
        print(f"Диапазон: от {group_min} до {group_max} МэВ")

        display(group[["File", "Median Energy", "Total Integral", "Total"]].sort_values("Median Energy"))

    return df_clean

def check_grouping_completeness(df_clean: pd.DataFrame) -> None:

    total_with_energy = df_clean.shape[0]
    total_grouped = df_clean[df_clean["EnergyGroup"] != -1].shape[0]

    if total_with_energy == total_grouped:
        print("\n Каждая строчка с указанной энергией отнесена к какой-либо группе.")
    else:
        print(f"\n Внимание: {total_with_energy - total_grouped} строк не были отнесены ни к одной группе.")

def plot_energy_groups(df_clean: pd.DataFrame, n_groups: int) -> None:
    if "EnergyGroup" not in df_clean.columns:
        print("Сначала нужно выполнить кластеризацию и присвоить группы.")
        return

    plt.figure(figsize=(10, 6))
    markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'x']  # добавлены маркеры на случай >4 групп

    for i in range(n_groups):
        group = df_clean[df_clean["EnergyGroup"] == i]
        if group.empty:
            continue
        plt.scatter(group["Median Energy"], group["Total Integral"],
                    label=f"Группа {i+1}",
                    marker=markers[i % len(markers)],
                    alpha=0.7)

    plt.xlabel("Median Energy (MeV)")
    plt.ylabel("Total Integral")
    plt.title("Группировка по энергии")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def calculate_group_summary(df_clean: pd.DataFrame, n_groups: int) -> pd.DataFrame:
    if "EnergyGroup" not in df_clean.columns:
        print("Сначала запустите ячейку с кластеризацией.")
        return pd.DataFrame()
    
    result = []

    for i in range(n_groups):
        group = df_clean[df_clean["EnergyGroup"] == i]

        total_integral = group["Total Integral"].sum()
        total_events = group["Total"].sum()

        if total_integral == 0:
            weighted_energy = None
        else:
            weighted_energy = (group["Median Energy"] * group["Total Integral"]).sum() / total_integral

        result.append({
            "Группа": i + 1,
            "Суммарный интеграл": round(total_integral, 2),
            "Взвешенная средняя энергия": round(weighted_energy, 3) if weighted_energy is not None else "н/д",
            "Количество событий": int(total_events)
        })

    summary_df = pd.DataFrame(result)
    display(summary_df)
    return summary_df

def sum_integral_and_ion_count(df_clean: pd.DataFrame,
                               ion_charge: float,
                               n_groups: int) -> pd.DataFrame:
    elementary_charge = 1.602e-19
    micro_to_coulomb = 1e-6

    if "EnergyGroup" not in df_clean.columns:
        print("Сначала запустите кластеризацию.")
        return pd.DataFrame()
    
    result = []

    for i in range(n_groups):
        group = df_clean[df_clean["EnergyGroup"] == i]
        total_integral = group["Total Integral"].sum()
        Q_coulomb = total_integral * micro_to_coulomb

        if Q_coulomb == 0:
            ion_count = 0
        else:
            ion_count = Q_coulomb / (ion_charge * elementary_charge)

        result.append({
            "Группа": i + 1,
            "Суммарный интеграл (мкКл)": round(total_integral, 2),
            "Количество ионов": f"{ion_count:.2e}"
        })

    ion_df = pd.DataFrame(result)
    display(ion_df)
    return ion_df

def calculate_cross_sections(df_clean: pd.DataFrame, n_groups: int, t: float = 7.26e17, 
                              eps: float = 0.02, ion_charge: float = 10.0, 
                              e_charge: float = 1.602e-19, to_pb: float = 1e36) -> pd.DataFrame:
    if "EnergyGroup" not in df_clean.columns:
        print("Сначала нужно провести кластеризацию.")
        return pd.DataFrame()

    result = []

    for i in range(n_groups):
        group = df_clean[df_clean["EnergyGroup"] == i]

        N_events = group["Total"].sum()
        total_integral = group["Total Integral"].sum()  # мкКл
        Q_coulomb = total_integral * 1e-6
        i_ions = Q_coulomb / (ion_charge * e_charge)

        if i_ions == 0:
            sigma_cm2 = 0
        else:
            sigma_cm2 = N_events / (i_ions * t * eps)

        sigma_pb = sigma_cm2 * to_pb

        result.append({
            "Группа": i + 1,
            "Событий N": int(N_events),
            "Ионов i": f"{i_ions:.2e}",
            "Сечение (см²)": f"{sigma_cm2:.3e}",
            "Сечение (пб)": round(sigma_pb, 2)
        })

    sigma_df = pd.DataFrame(result)
    display(sigma_df)
    return sigma_df

def plot_cross_section_vs_energy(sigma_df: pd.DataFrame, summary_df: pd.DataFrame):
    if sigma_df is None or summary_df is None:
        print("Ошибка: одна из таблиц не передана.")
        return

    merged_df = pd.merge(sigma_df, summary_df, on="Группа")

    plt.figure(figsize=(8, 6))
    plt.plot(merged_df["Взвешенная средняя энергия"], merged_df["Сечение (пб)"],
             marker='o', linestyle='-', markersize=8)

    plt.xlabel("Взвешенная средняя энергия (МэВ)")
    plt.ylabel("Сечение (пб)")
    plt.title("")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def calculate_cross_section_uncertainty(sigma_df: pd.DataFrame) -> pd.DataFrame:
    if sigma_df is None or sigma_df.empty:
        print("Ошибка: таблица sigma_df пуста или не передана.")
        return None

    try:
        sigma_df["Сечение (пб)"] = pd.to_numeric(sigma_df["Сечение (пб)"], errors='coerce')
        sigma_df["Событий N"] = pd.to_numeric(sigma_df["Событий N"], errors='coerce')

        N = sigma_df["Событий N"]
        sigma_pb = sigma_df["Сечение (пб)"]

        sigma_up = (N + (1 + N ** 0.5)) * sigma_pb / N
        sigma_err_up = sigma_up - sigma_pb

        sigma_low = (N - N ** 0.5) * sigma_pb / N
        sigma_err_low = sigma_pb - sigma_low

        sigma_df["Погрешность вверх (пб)"] = sigma_err_up.round(2)
        sigma_df["Погрешность вниз (пб)"] = sigma_err_low.round(2)

        display(sigma_df[["Группа", "Сечение (пб)", "Погрешность вверх (пб)", "Погрешность вниз (пб)"]])
        return sigma_df

    except Exception as e:
        print(f"Произошла ошибка при расчёте: {e}")
        return None

def plot_cross_section_with_errors(summary_df: pd.DataFrame, sigma_df: pd.DataFrame):
    if summary_df is None or sigma_df is None:
        print("Ошибка: одна из таблиц не определена.")
        return

    try:
        merged_df = pd.merge(sigma_df, summary_df, on="Группа")

        x = merged_df["Взвешенная средняя энергия"]
        y = merged_df["Сечение (пб)"]
        yerr_lower = merged_df["Погрешность вниз (пб)"]
        yerr_upper = merged_df["Погрешность вверх (пб)"]

        plt.figure(figsize=(8, 6))
        plt.errorbar(x, y,
                     yerr=[yerr_lower, yerr_upper],
                     fmt='o-', capsize=5, markersize=8)

        plt.xlabel("Взвешенная средняя энергия (МэВ)")
        plt.ylabel("Сечение (пб)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Произошла ошибка при построении графика: {e}")

def plot_cross_section_with_energy_errors(df_clean: pd.DataFrame,
                                          summary_df: pd.DataFrame,
                                          sigma_df: pd.DataFrame,
                                          n_groups: int):
    if any(df is None for df in [df_clean, summary_df, sigma_df]):
        print("Ошибка: одна или несколько таблиц не определены.")
        return

    try:
        energy_ranges = {}

        for i in range(n_groups):
            group = df_clean[df_clean["EnergyGroup"] == i]
            if group.empty:
                continue
            group_min = group["Energy"].min()
            group_max = group["Energy"].max()
            group_center = group["Energy"].mean()

            energy_ranges[i + 1] = {
                "min": group_min,
                "max": group_max,
                "center": group_center
            }

        merged_df = pd.merge(sigma_df, summary_df, on="Группа")

        x = merged_df["Взвешенная средняя энергия"]
        y = merged_df["Сечение (пб)"]
        yerr_lower = merged_df["Погрешность вниз (пб)"]
        yerr_upper = merged_df["Погрешность вверх (пб)"]
        xerr_lower = []
        xerr_upper = []

        for _, row in merged_df.iterrows():
            group_id = row["Группа"]
            E_center = row["Взвешенная средняя энергия"]
            E_min = energy_ranges[group_id]["min"]
            E_max = energy_ranges[group_id]["max"]
            xerr_lower.append(E_center - E_min)
            xerr_upper.append(E_max - E_center)

        plt.figure(figsize=(8, 6))
        plt.errorbar(x, y,
                     xerr=[xerr_lower, xerr_upper],
                     yerr=[yerr_lower, yerr_upper],
                     fmt='o-', capsize=5, markersize=8)

        plt.xlabel("Взвешенная средняя энергия (МэВ)")
        plt.ylabel("Сечение (пб)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Ошибка при построении графика: {e}")

def build_cross_section_table(df_clean: pd.DataFrame, summary_df: pd.DataFrame,
                               sigma_df: pd.DataFrame, n_groups: int) -> pd.DataFrame:
    if any(df is None for df in [df_clean, summary_df, sigma_df]):
        print("Ошибка: отсутствуют необходимые таблицы.")
        return pd.DataFrame()

    energy_ranges = {}
    for i in range(n_groups):
        group = df_clean[df_clean["EnergyGroup"] == i]
        if group.empty:
            continue
        energy_ranges[i + 1] = {
            "min": group["Energy"].min(),
            "max": group["Energy"].max(),
            "center": group["Energy"].mean()
        }

    merged_df = pd.merge(sigma_df, summary_df, on="Группа")

    rows = []
    for _, row in merged_df.iterrows():
        group = int(row["Группа"])
        E = row["Взвешенная средняя энергия"]
        E_min = energy_ranges[group]["min"]
        E_max = energy_ranges[group]["max"]
        E_err_plus = round(E_max - E, 3)
        E_err_minus = round(E - E_min, 3)

        sigma = round(row["Сечение (пб)"], 2)
        sigma_err_plus = round(row["Погрешность вверх (пб)"], 2)
        sigma_err_minus = round(row["Погрешность вниз (пб)"], 2)

        rows.append({
            "Точка": group,
            "Энергия (МэВ)": round(E, 3),
            "Ошибка энергии +": E_err_plus,
            "Ошибка энергии –": E_err_minus,
            "Сечение (пб)": sigma,
            "Ошибка сечения +": sigma_err_plus,
            "Ошибка сечения –": sigma_err_minus
        })

    final_table = pd.DataFrame(rows)
    display(final_table)
    return final_table

def build_group_summary(df_clean: pd.DataFrame,
                        summary_df: pd.DataFrame,
                        n_groups: int) -> pd.DataFrame:
    if df_clean is None or summary_df is None:
        print("Ошибка: не найдены нужные данные.")
        return pd.DataFrame()

    group_info = []

    for i in range(n_groups):
        group_data = df_clean[df_clean["EnergyGroup"] == i]
        if group_data.empty:
            continue

        group_id   = i + 1
        group_min  = group_data["Energy"].min()
        group_max  = group_data["Energy"].max()
        group_mean = group_data["Energy"].mean()
        weighted_row = summary_df[summary_df["Группа"] == group_id]
        if not weighted_row.empty:
            weighted_energy = weighted_row["Взвешенная средняя энергия"].values[0]
        else:
            weighted_energy = None

        err_plus  = round(group_max - weighted_energy, 3)
        err_minus = round(weighted_energy - group_min, 3)

        group_info.append({
            "Группа": group_id,
            "Центральная энергия (средняя)": round(group_mean, 3),
            "Взвешенная энергия": round(weighted_energy, 3),
            "Диапазон (Мин–Макс)": f"{round(group_min, 2)} – {round(group_max, 2)}",
            "Ошибка + (от макс)": err_plus,
            "Ошибка – (от мин)": err_minus
        })

    group_summary_df = pd.DataFrame(group_info)
    display(group_summary_df)
    return group_summary_df

def plot_shifted_cross_section(df_clean: pd.DataFrame,
                               summary_df: pd.DataFrame,
                               sigma_df: pd.DataFrame,
                               n_groups: int,
                               substrate_loss: float = 3.9,
                               target_loss: float = 1.08) -> tuple:

    if df_clean is None or summary_df is None or sigma_df is None:
        print("Ошибка: не найдены необходимые таблицы.")
        return None

    energy_shift = substrate_loss + target_loss / 2
    print(f"Смещение энергии (подложка + мишень/2): {energy_shift:.3f} МэВ")

    energy_ranges = {}
    for i in range(n_groups):
        group = df_clean[df_clean["EnergyGroup"] == i]
        if group.empty:
            continue
        energy_ranges[i + 1] = {
            "min": group["Energy"].min(),
            "max": group["Energy"].max()
        }

    merged_df = pd.merge(sigma_df, summary_df, on="Группа")

    x_lab = merged_df["Взвешенная средняя энергия"] - energy_shift
    y = merged_df["Сечение (пб)"]
    yerr_lower = merged_df["Погрешность вниз (пб)"]
    yerr_upper = merged_df["Погрешность вверх (пб)"]

    xerr_lower = []
    xerr_upper = []

    for _, row in merged_df.iterrows():
        group_id = row["Группа"]
        E_center = row["Взвешенная средняя энергия"]
        E_min = energy_ranges[group_id]["min"]
        E_max = energy_ranges[group_id]["max"]
        xerr_lower.append(E_center - E_min)
        xerr_upper.append(E_max - E_center)

    # Построение графика
    plt.figure(figsize=(8, 6))
    plt.errorbar(x_lab, y,
                 xerr=[xerr_lower, xerr_upper],
                 yerr=[yerr_lower, yerr_upper],
                 fmt='o-', capsize=5, markersize=8)

    plt.xlabel("Энергия в лабораторной СК (МэВ)")
    plt.ylabel("Сечение (пб)")
    plt.title("")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return x_lab, y, xerr_lower, xerr_upper, yerr_lower, yerr_upper

def plot_single_channel(data, label="Канал", color="blue", marker='o',
                        title="Сечение реакции", xlabel="Энергия (МэВ)", ylabel="Сечение (пб)") -> None:
    df = pd.DataFrame(data, columns=[
        "Energy", "Value", "Err_Up_Y", "Err_Down_Y", "Err_Down_X", "Err_Up_X"
    ])

    df["Energy"] = df["Energy"] / 1e6            # эВ → МэВ
    df["Value"] = df["Value"] * 1e12             # барн → пб
    df["Err_Up_Y"] = df["Err_Up_Y"] * 1e12
    df["Err_Down_Y"] = df["Err_Down_Y"] * 1e12
    df["Err_Down_X"] = df["Err_Down_X"] / 1e6
    df["Err_Up_X"] = df["Err_Up_X"] / 1e6

    # Построение графика
    plt.figure(figsize=(8, 6))
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
data_4n = [
    (1.2180E+08, 5.0000E-11, 3.0000E-11, 2.0000E-11, 1.2180E+06, 1.2180E+06),
    (1.2810E+08, 1.7000E-10, 8.0000E-11, 5.0000E-11, 1.2810E+06, 1.2810E+06),
    (1.3300E+08, 1.8000E-10, 8.0000E-11, 6.0000E-11, 1.3300E+06, 1.3300E+06)
]

def plot_combined_cross_section(x_lab, y, xerr_lower, xerr_upper, yerr_lower, yerr_upper,
                                 A1=26, A2=238, Q=-78.2) -> None:
    if any(arg is None for arg in [x_lab, y, xerr_lower, xerr_upper, yerr_lower, yerr_upper]):
        print("Ошибка: не все входные данные заданы.")
        return

    excitation_energy = x_lab * (A2 / (A1 + A2)) + Q

    data_4n = [
        (1.2180E+08, 5.0000E-11, 3.0000E-11, 2.0000E-11, 1.2180E+06, 1.2180E+06),
        (1.2810E+08, 1.7000E-10, 8.0000E-11, 5.0000E-11, 1.2810E+06, 1.2810E+06),
        (1.3300E+08, 1.8000E-10, 8.0000E-11, 6.0000E-11, 1.3300E+06, 1.3300E+06)
    ]
    
    df_gate = pd.DataFrame(data_4n, columns=[
        "Energy", "Value", "Err_Up_Y", "Err_Down_Y", "Err_Down_X", "Err_Up_X"
    ])
    
    df_gate["Energy"] = df_gate["Energy"] / 1e6  # эВ → МэВ
    df_gate["Value"] = df_gate["Value"] * 1e12  # барн → пб
    df_gate["Err_Up_Y"] = df_gate["Err_Up_Y"] * 1e12
    df_gate["Err_Down_Y"] = df_gate["Err_Down_Y"] * 1e12
    df_gate["Err_Down_X"] = df_gate["Err_Down_X"] / 1e6
    df_gate["Err_Up_X"] = df_gate["Err_Up_X"] / 1e6

    fig, ax1 = plt.subplots(figsize=(9, 6))

    ax1.errorbar(x_lab, y,
                 xerr=[xerr_lower, xerr_upper],
                 yerr=[yerr_lower, yerr_upper],
                 fmt='o-', capsize=5, markersize=8, label="FLNR, SHELS", color='green')

    ax1.errorbar(df_gate["Energy"], df_gate["Value"],
                 xerr=[df_gate["Err_Down_X"], df_gate["Err_Up_X"]],
                 yerr=[df_gate["Err_Down_Y"], df_gate["Err_Up_Y"]],
                 fmt='s', capsize=5, label="4n (Rf-260), Gates", color='gold')

    ax1.set_xlabel("Энергия в лабораторной СК (МэВ)")
    ax1.set_ylabel("Сечение (пб)")
    ax1.grid(True)
    ax1.legend()

    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim()) 
    ax2.set_xticks(x_lab)
    ax2.set_xticklabels([f"{val:.1f}" for val in excitation_energy])
    ax2.set_xlabel("Энергия возбуждения (МэВ)")

    plt.title("")
    plt.tight_layout()
    plt.show()