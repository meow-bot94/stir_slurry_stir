from dataclasses import dataclass, field
from typing import List
import numpy as np
from src.dp_model.initial_dp_tank_state import InitialDPTankState, default_dp_tank_state
from src.dp_model.tank_state import TankState
from src.scenario.scenario_config import ScenarioConfig, default_scenario_config


@dataclass
class DPModelConfig:
    scenario: ScenarioConfig
    active_tank_index: int  # zero-indexed idx of tank that is currently active. Must be valid
    initial_tank_status: List[InitialDPTankState]

    slurry_unit_violate_penalty: float
    slurry_unit_wastage_penalty: float

    def __post_init__(self):
        assert 0 <= self.active_tank_index < len(self.initial_tank_status)

    def init_tank_states(self) -> List[TankState]:
        states_list = list()
        for idx, config in enumerate(self.initial_tank_status):
            tank_state = TankState(
                idx,
                config.current_slurry_unit,
                config.cleaning_period_remaining,
                config.filling_period_remaining,
                np.inf,
            )
            states_list.append(tank_state)
        return states_list


default_dp_model_config = DPModelConfig(
    scenario=default_scenario_config,
    active_tank_index=0,
    initial_tank_status=[default_dp_tank_state]*3,
    slurry_unit_violate_penalty=1000,
    slurry_unit_wastage_penalty=50,
)
