from dataclasses import dataclass, field
from typing import List

from tank_state import TankState, default_tank_state


@dataclass
class ModelConfig:
    num_tanks: int
    tank_status: List[TankState]
    tank_fill_capacity_kg: float
    tank_min_capacity_kg: float

    tank_clean_minutes: int
    tank_fill_minutes: int

    lookahead_minutes: int
    bin_minutes: int
    rand_seed: int

    min_wip_cluster: int
    max_wip_cluster: int
    max_wip_magnitude: float
    wafer_slurry_consumption_kg: float

    tank_clean_period: int = field(init=False)
    tank_fill_period: int = field(init=False)
    lookahead_period: int = field(init=False)

    def __post_init__(self):
        self.tank_clean_period = int(self.tank_clean_minutes / self.bin_minutes)
        self.tank_fill_period = int(self.tank_fill_minutes / self.bin_minutes)
        self.lookahead_period = int(self.lookahead_minutes / self.bin_minutes)
        self.tank_status = sorted(self.tank_status)


default_config = ModelConfig(
    num_tanks=3,
    tank_status=[default_tank_state]*3,
    tank_fill_capacity_kg=200,
    tank_min_capacity_kg=60,
    min_wip_cluster=1,
    max_wip_cluster=4,
    max_wip_magnitude=200,
    wafer_slurry_consumption_kg=3,
    tank_clean_minutes=60,
    tank_fill_minutes=90,
    lookahead_minutes=60*8,
    bin_minutes=10,
    rand_seed=7,
)
