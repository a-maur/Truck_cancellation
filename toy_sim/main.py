import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.model_selection import train_test_split

from config import PARCEL_TYPES, SORTING_CENTERS, SimulationConfig, build_day_configuration, validate_shapes
from core import N_HOURS, create_macro_stat_dataset_all_origins, generate_raw_data


def build_raw_save_columns() -> list[str]:
    save_columns = ["center", "day", "season"]
    save_columns += [f"vol_h{hour}" for hour in range(N_HOURS)]

    destination_suffixes = [
        "_n_exp_trucks",
        "_frac_last_truck_needed",
        "_hist_avg_vol",
        "_hist_std_vol",
        "_last_truck_needed",
    ]

    for dest in SORTING_CENTERS:
        for parcel_type in PARCEL_TYPES:
            save_columns.append(f"{dest}_{parcel_type}")
        for suffix in destination_suffixes:
            save_columns.append(f"{dest}{suffix}")

    return save_columns


def run(cfg: SimulationConfig | None = None, correlation_dest_list: Iterable[float] | None = None) -> None:
    cfg = cfg or SimulationConfig()
    if correlation_dest_list is not None:
        cfg.correlation_dest_list = list(correlation_dest_list)
    validate_shapes()

    output_dir = Path(__file__).resolve().parent / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dict = build_day_configuration()
    save_columns = build_raw_save_columns()

    for correlation_dest in cfg.correlation_dest_list:
        df_raw = generate_raw_data(
            data_dict=data_dict,
            corr_dest=correlation_dest,
            corr_type=cfg.correlation_type,
            n_parcels_per_truck=cfg.n_parcels_per_truck,
            n_weeks=cfg.n_weeks,
            n_weeks_high_season=cfg.n_weeks_high_season,
            margin=cfg.margin,
        )

        if cfg.save_to_file:
            df_raw[save_columns].to_pickle(output_dir / "df_raw.pkl")

        df_indices = np.arange(df_raw.shape[0])
        idx_train, idx_test = train_test_split(
            df_indices,
            test_size=cfg.train_test_ratio,
            random_state=cfg.random_seed,
            shuffle=True,
        )

        train_df_raw = df_raw.iloc[idx_train]
        test_df_raw = df_raw.iloc[idx_test]
        train_df = create_macro_stat_dataset_all_origins(train_df_raw)
        test_df = create_macro_stat_dataset_all_origins(test_df_raw)

        if cfg.save_to_file:
            train_df.to_pickle(output_dir / "df_per_dest_train.pkl")
            test_df.to_pickle(output_dir / "df_per_dest_test.pkl")


def main(correlation_dest_list: Iterable[float] | None = None) -> None:
    if correlation_dest_list is None:
        parser = argparse.ArgumentParser(description="Run truck cancellation data simulation.")
        parser.add_argument(
            "correlation_dest",
            nargs="?",
            type=float,
            help="Single destination-correlation value, e.g. 0.9",
        )
        args = parser.parse_args()
        if args.correlation_dest is not None:
            correlation_dest_list = [args.correlation_dest]

    run(correlation_dest_list=correlation_dest_list)


if __name__ == "__main__":
    main()
