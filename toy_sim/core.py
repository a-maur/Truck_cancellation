import numpy as np
import pandas as pd

from config import (
    AVERAGE_VOL_DEST,
    HOUR_EVOLUTION_DICT,
    PARCEL_TYPES,
    SORTING_CENTERS,
)

# Core simulation and feature-engineering utilities used by main.py.

N_DEST = len(SORTING_CENTERS) - 1
N_TYPES = len(PARCEL_TYPES)
N_HOURS = max(len(hour_ranges) for hour_ranges in HOUR_EVOLUTION_DICT.values()) + 1


def destinations_for_origin(origin: str) -> list[str]:
    """Return all destination centers reachable from an origin (exclude self)."""
    return [center for center in SORTING_CENTERS if center != origin]


def build_average_vol_by_origin() -> dict[str, np.ndarray]:
    """Split baseline lane volume into parcel-type volumes for each origin."""
    average_vol_by_origin: dict[str, np.ndarray] = {}
    for i_origin, origin in enumerate(SORTING_CENTERS):
        avg_dest = AVERAGE_VOL_DEST[i_origin]
        average_vol_type_a = np.array(avg_dest) * 0.3
        average_vol_type_b = avg_dest - average_vol_type_a
        average_vol_by_origin[origin] = np.stack([average_vol_type_a, average_vol_type_b], axis=-1)
    return average_vol_by_origin


AVERAGE_VOL_BY_ORIGIN = build_average_vol_by_origin()


def uniform_corr_matrix(size: int, rho: float) -> np.ndarray:
    """Create a constant-correlation matrix with unit diagonal."""
    corr = np.full((size, size), rho)
    np.fill_diagonal(corr, 1.0)
    return corr


def create_partially_correlated_dataset(
    mu: np.ndarray,
    sigma: np.ndarray,
    corr_dest: np.ndarray,
    corr_type: np.ndarray,
    n_days: int = 10_000,
) -> np.ndarray:
    """Sample correlated daily parcel counts across (destination x parcel type)."""
    corr = np.kron(corr_dest, corr_type)
    cov = np.diag(sigma) @ corr @ np.diag(sigma)
    samples = np.random.multivariate_normal(mean=mu, cov=cov, size=n_days).astype(int)
    return samples


def build_dataframe(
    origin: str,
    corr_dest: float,
    corr_type: float,
    day_dict: dict,
    n_days: int = 10_000,
    n_parcels_per_truck: int = 100,
) -> pd.DataFrame:
    """Build per-origin daily dataframe with lane totals and expected trucks."""
    columns = []
    for dest in destinations_for_origin(origin):
        for parcel_type in PARCEL_TYPES:
            columns.append(f"{dest}_{parcel_type}")

    corr_dest_matrix = uniform_corr_matrix(N_DEST, corr_dest)
    corr_type_matrix = uniform_corr_matrix(N_TYPES, corr_type)

    avg_origin = AVERAGE_VOL_BY_ORIGIN[origin]
    avg_flat = avg_origin.reshape(-1)
    sigma = 0.1 * avg_flat
    scaled_mean = avg_flat * day_dict["scaling_factor"]

    data_array_flat = create_partially_correlated_dataset(
        scaled_mean, sigma, corr_dest_matrix, corr_type_matrix, n_days=n_days
    )
    data_array = data_array_flat.reshape(len(data_array_flat), N_DEST, N_TYPES)
    df = pd.DataFrame(data_array_flat, columns=columns)

    day_total = 0
    for i_dest, dest in enumerate(destinations_for_origin(origin)):
        total_dest = data_array[:, i_dest, :].sum(axis=1)
        df[f"{dest}_total"] = total_dest
        day_total += total_dest
        n_exp_trucks = np.ceil(np.percentile(total_dest, 90) / n_parcels_per_truck)
        df[f"{dest}_n_exp_trucks"] = n_exp_trucks

    for i_parcel, parcel_type in enumerate(PARCEL_TYPES):
        total_parcel_type = data_array[:, :, i_parcel].sum(axis=1)
        df[f"{parcel_type}_total"] = total_parcel_type

    day_of_the_week = np.arange(n_days) % len(day_dict["days"]) + min(day_dict["days"])
    df["day_total"] = day_total
    df["day"] = day_of_the_week
    df["season"] = day_dict["season"]
    return df


def merge_df_all_days(
    dfs: list[pd.DataFrame], days: list[list[int]], shuffle: bool = False, random_state: int | None = None
) -> pd.DataFrame | None:
    """Interleave day-group dataframes in round-robin chunks."""
    if len(dfs) != len(days):
        raise ValueError("dfs and days must have the same length")
    if not dfs:
        return None
    if dfs[0] is None:
        return None

    if shuffle:
        dfs = [df.sample(frac=1, random_state=random_state).reset_index(drop=True) for df in dfs]

    pointers = [0] * len(dfs)
    quotas = [len(day_list) for day_list in days]
    out = []

    while True:
        for i, quota in enumerate(quotas):
            if pointers[i] + quota > len(dfs[i]):
                return pd.DataFrame(out)
        for i, quota in enumerate(quotas):
            rows = dfs[i].iloc[pointers[i] : pointers[i] + quota]
            out.extend(rows.to_dict("records"))
            pointers[i] += quota


def compute_last_truck_overflow_and_historical_averages(
    origin: str, df: pd.DataFrame, n_parcels_per_truck: int = 100, margin: float = 0.2
) -> pd.DataFrame:
    """Compute overflow carry-over and historical statistics per destination/day."""
    for dest in destinations_for_origin(origin):
        df[f"{dest}_total2"] = df[f"{dest}_total"]
        df[f"{dest}_overflow"] = np.nan
        df[f"{dest}_last_truck_needed"] = False

    for index, _row in df.iterrows():
        for dest in destinations_for_origin(origin):
            if index > 0:
                df.at[index, f"{dest}_total2"] += df.at[index - 1, f"{dest}_overflow"]

            total_dest = df.at[index, f"{dest}_total2"]
            n_exp_trucks = df.at[index, f"{dest}_n_exp_trucks"]
            df.at[index, f"{dest}_last_truck_needed"] = (
                total_dest > (n_exp_trucks - (1 - margin)) * n_parcels_per_truck
            )

            overflow = np.maximum(
                np.maximum(total_dest - n_exp_trucks * n_parcels_per_truck, 0),
                np.where(
                    total_dest < (n_exp_trucks - (1 - margin)) * n_parcels_per_truck,
                    np.maximum(total_dest - (n_exp_trucks - 1) * n_parcels_per_truck, 0),
                    0,
                ),
            )
            df.at[index, f"{dest}_overflow"] = overflow

    day_total2 = 0
    total_overflow = 0
    for dest in destinations_for_origin(origin):
        df[f"{dest}_frac_last_truck_needed"] = df.groupby("day")[f"{dest}_last_truck_needed"].transform("mean")
        df[f"{dest}_hist_avg_vol"] = df.groupby("day")[f"{dest}_total2"].transform("mean")
        df[f"{dest}_hist_std_vol"] = df.groupby("day")[f"{dest}_total2"].transform("std")
        day_total2 += df[f"{dest}_total2"]
        total_overflow += df[f"{dest}_overflow"]

    df["day_total2"] = day_total2
    df["total_overflow"] = total_overflow
    df["hist_avg_vol_tot"] = df.groupby("day")["day_total2"].transform("mean")
    df["hist_std_vol_tot"] = df.groupby("day")["day_total2"].transform("std")

    return df.iloc[1:]


def calculate_hourly_volumes(n_days: int) -> list[np.ndarray]:
    """Sample intraday cumulative volume profiles for each parcel type."""
    rng = np.random.default_rng()
    hour_evolution = []

    for ranges_ in HOUR_EVOLUTION_DICT.values():
        rnd_array = np.vstack([rng.uniform(min_v, max_v, size=n_days) for min_v, max_v in ranges_]).T
        rnd_array = np.column_stack((rnd_array, np.ones((n_days, 1))))
        hour_evolution.append(rnd_array)

    max_len = max(he.shape[1] for he in hour_evolution)
    for i, he in enumerate(hour_evolution):
        if he.shape[1] < max_len:
            n_missing = max_len - he.shape[1]
            hour_evolution[i] = np.column_stack((he, np.ones((n_days, n_missing))))

    return hour_evolution


def add_hourly_volumes(df: pd.DataFrame, hour_evolution: list[np.ndarray]) -> pd.DataFrame:
    """Add cumulative hourly volume columns (`vol_h*`) to the daily dataframe."""
    day_volume = (df["day_total2"].values - df["day_total"].values)[:, None]
    for i, he in enumerate(hour_evolution):
        day_volume = day_volume + (df[f"{PARCEL_TYPES[i]}_total"].values[:, None] * he).astype(int)

    columns_hour = [f"vol_h{hour}" for hour in range(N_HOURS)]
    df[columns_hour] = day_volume
    return df


def create_macro_stat_dataset(df: pd.DataFrame, origin: str) -> pd.DataFrame:
    """Expand one-origin daily data into per-hour, per-destination training rows."""
    columns_stat_hour = ["hour", "min", "max", "mean", "std", "delta"]
    columns_day = ["center", "day", "season", "hist_avg_vol_tot", "hist_std_vol_tot"]
    columns_dest = ["n_exp_trucks", "frac_last_truck_needed", "hist_avg_vol", "hist_std_vol", "last_truck_needed"]
    col_vol_hour = [f"vol_h{hour}" for hour in range(N_HOURS)]
    day_volume = df[col_vol_hour].values
    df_stat = []

    for hour in range(1, N_HOURS):
        subset = day_volume[:, 0 : hour + 1]
        min_vals = np.min(subset, axis=1).astype(int)
        max_vals = np.max(subset, axis=1).astype(int)
        mean_vals = np.mean(subset, axis=1).astype(int)
        delta_vals = (subset[:, -1] - subset[:, -2]).astype(int)
        std_vals = np.std(subset, axis=1).astype(int)

        stats = np.column_stack(
            (
                np.full_like(min_vals, hour),
                min_vals,
                max_vals,
                mean_vals,
                std_vals,
                delta_vals,
            )
        )
        df_hour = pd.DataFrame(stats, columns=columns_stat_hour)
        df_hour[columns_day] = df[columns_day].reset_index(drop=True)

        for dest in destinations_for_origin(origin):
            df_new = df_hour.copy()
            df_new["dest"] = dest
            per_dest_columns = [f"{dest}_{col}" for col in columns_dest]
            df_new[columns_dest] = df[per_dest_columns].reset_index(drop=True)
            df_stat.append(df_new)

    return pd.concat(df_stat, ignore_index=True)


def generate_data_single_origin(
    origin: str,
    data_dict: dict,
    corr_dest: float,
    corr_type: float,
    n_parcels_per_truck: int = 100,
    n_weeks: int = 52,
    n_weeks_high_season: int = 12,
    margin: float = 0.2,
    shuffle: bool = False,
) -> pd.DataFrame:
    """Generate full daily dataset for one origin across all seasons/day groups."""
    df_list = []
    for season_name, season_dict in data_dict.items():
        df_day_list = []
        for day_name, day_dict in season_dict.items():
            if season_name == "low_season":
                n_days = n_weeks * len(day_dict["days"])
            elif season_name == "high_season":
                n_days = n_weeks_high_season * len(day_dict["days"])
            else:
                raise ValueError("data_dict key must be 'low_season' or 'high_season'")

            if n_days == 0:
                continue

            print(season_name, day_name)
            df_day = build_dataframe(
                origin,
                corr_dest,
                corr_type,
                day_dict,
                n_days=n_days,
                n_parcels_per_truck=n_parcels_per_truck,
            )
            df_day_list.append(df_day)
            data_dict[season_name][day_name]["dataframe"] = df_day

        if not df_day_list:
            continue

        merged = merge_df_all_days(df_day_list, [day["days"] for day in season_dict.values()])
        merged = compute_last_truck_overflow_and_historical_averages(
            origin, merged, n_parcels_per_truck=n_parcels_per_truck, margin=margin
        )
        df_list.append(merged)

    df = pd.concat(df_list, ignore_index=True)
    df["center"] = origin
    if shuffle:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def generate_raw_data(
    data_dict: dict,
    corr_dest: float,
    corr_type: float,
    n_parcels_per_truck: int = 100,
    n_weeks: int = 52,
    n_weeks_high_season: int = 12,
    margin: float = 0.2,
) -> pd.DataFrame:
    """Generate and merge raw data for all origins, then shuffle rows."""
    df_all_origins = []
    for origin in SORTING_CENTERS:
        df_origin = generate_data_single_origin(
            origin,
            data_dict,
            corr_dest,
            corr_type,
            n_parcels_per_truck=n_parcels_per_truck,
            margin=margin,
            n_weeks=n_weeks,
            n_weeks_high_season=n_weeks_high_season,
        )
        n_days = df_origin.shape[0]
        daily_volumes = calculate_hourly_volumes(n_days)
        df_origin = add_hourly_volumes(df_origin, daily_volumes)
        df_all_origins.append(df_origin)

    df = pd.concat(df_all_origins, ignore_index=True)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def create_macro_stat_dataset_all_origins(df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-destination macro-stat dataset for each origin and concatenate."""
    df_all_origins = []
    for origin in SORTING_CENTERS:
        df_origin = df[df["center"] == origin]
        if len(df_origin) == 0:
            continue
        df_stat = create_macro_stat_dataset(df_origin, origin)
        df_all_origins.append(df_stat)
    return pd.concat(df_all_origins, ignore_index=True)
