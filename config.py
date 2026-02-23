from dataclasses import dataclass, field

import numpy as np


SORTING_CENTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
PARCEL_TYPES = ["typeA", "typeB"]


AVERAGE_VOL_DEST = np.array(
    [
        [220, 350, 400, 610, 500, 390, 530, 280, 310],
        [510, 690, 880, 440, 360, 570, 630, 310, 280],
        [230, 560, 380, 410, 350, 440, 660, 550, 480],
        [230, 440, 380, 410, 370, 470, 430, 550, 440],
        [440, 510, 310, 410, 320, 470, 500, 450, 450],
        [500, 590, 410, 420, 330, 370, 750, 450, 530],
        [280, 360, 480, 430, 330, 510, 800, 410, 380],
        [230, 470, 670, 440, 380, 550, 450, 410, 380],
        [280, 580, 680, 450, 390, 500, 290, 550, 300],
        [280, 500, 560, 460, 300, 500, 400, 650, 280],
    ]
)


HOUR_EVOLUTION_DICT = {
    "typeA": [[0.40, 0.70], [0.70, 0.90]],
    "typeB": [[0.40, 0.60], [0.60, 0.70], [0.70, 0.80], [0.80, 0.90]],
}


@dataclass
class SimulationConfig:
    n_weeks: int = 100
    n_weeks_high_season: int = 10
    n_parcels_per_truck: int = 100
    margin: float = 0.0
    correlation_type: float = 0.3
    correlation_dest_list: list[float] = field(default_factory=lambda: [0.9])
    train_test_ratio: float = 0.2
    save_to_file: bool = True
    output_dir: str = "example_data"
    random_seed: int = 42


def validate_shapes() -> None:
    n_centers = len(SORTING_CENTERS)
    n_dest = n_centers - 1
    if AVERAGE_VOL_DEST.shape != (n_centers, n_dest):
        raise ValueError(f"average_vol_matrix must have shape {(n_centers, n_dest)}")


def build_day_configuration() -> dict:
    f_low_season = 1.0
    f_high_season = 1.5
    f_mon = 1.2
    f_tueth = 1.0
    f_fr = 1.1

    return {
        "low_season": {
            "Mon": {"scaling_factor": f_low_season * f_mon, "days": [0], "season": 0},
            "TueTh": {"scaling_factor": f_low_season * f_tueth, "days": [1, 2, 3], "season": 0},
            "Fr": {"scaling_factor": f_low_season * f_fr, "days": [4], "season": 0},
        },
        "high_season": {
            "Mon": {"scaling_factor": f_high_season * f_mon, "days": [0], "season": 1},
            "TueTh": {"scaling_factor": f_high_season * f_tueth, "days": [1, 2, 3], "season": 1},
            "Fr": {"scaling_factor": f_high_season * f_fr, "days": [4], "season": 1},
        },
    }
