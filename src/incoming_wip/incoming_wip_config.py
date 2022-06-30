from dataclasses import dataclass, field

from src.scenario.scenario_config import ScenarioConfig, default_scenario_config


@dataclass
class IncomingWipConfig:
    scenario: ScenarioConfig
    rand_seed: int

    min_wip_cluster: int
    max_wip_cluster: int
    max_wip_magnitude: float
    wafer_slurry_consumption_kg: float


default_incoming_wip_config = IncomingWipConfig(
    scenario=default_scenario_config,
    rand_seed=7,
    min_wip_cluster=1,
    max_wip_cluster=4,
    max_wip_magnitude=200,
    wafer_slurry_consumption_kg=3,
)

