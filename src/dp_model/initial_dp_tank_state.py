from __future__ import annotations
from dataclasses import dataclass, field

from src.scenario.scenario_config import ScenarioConfig, default_scenario_config


@dataclass
class InitialDPTankState:
    scenario: ScenarioConfig
    current_slurry_kg: float
    cleaning_minutes_remaining: int
    filling_minutes_remaining: int

    tank_fill_capacity_kg: float
    tank_min_capacity_kg: float
    tank_clean_minutes: int
    tank_fill_minutes: int

    current_slurry_unit: int = field(init=False)
    cleaning_period_remaining: int = field(init=False)
    filling_period_remaining: int = field(init=False)
    tank_fill_capacity_slurry_unit: int = field(init=False)
    tank_min_capacity_slurry_unit: int = field(init=False)
    tank_clean_period: int = field(init=False)
    tank_fill_period: int = field(init=False)

    def __post_init__(self):
        self.current_slurry_unit = round(self.current_slurry_kg / self.scenario.slurry_unit_bin_kg)
        self.cleaning_period_remaining = round(self.cleaning_minutes_remaining / self.scenario.period_bin_minutes)
        self.filling_period_remaining = round(self.filling_minutes_remaining / self.scenario.period_bin_minutes)
        self.tank_fill_capacity_slurry_unit = round(self.tank_fill_capacity_kg / self.scenario.slurry_unit_bin_kg)
        self.tank_min_capacity_slurry_unit = round(self.tank_min_capacity_kg / self.scenario.slurry_unit_bin_kg)
        self.tank_clean_period = round(self.tank_clean_minutes / self.scenario.period_bin_minutes)
        self.tank_fill_period = round(self.tank_fill_minutes / self.scenario.period_bin_minutes)
        self._verify_state()

    def _verify_state(self):
        if self.current_slurry_kg > 0:
            assert self.filling_minutes_remaining == 0
            assert self.cleaning_minutes_remaining == 0
        if self.cleaning_minutes_remaining > 0:
            assert self.filling_minutes_remaining > 0


default_dp_tank_state = InitialDPTankState(
    scenario=default_scenario_config,
    current_slurry_kg=200,
    cleaning_minutes_remaining=0,
    filling_minutes_remaining=0,
    tank_fill_capacity_kg=200,
    tank_min_capacity_kg=60,
    tank_clean_minutes=60,
    tank_fill_minutes=90,
)
