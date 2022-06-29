import pytest
from model_config import ModelConfig
from tank_state import default_tank_state


def test_model_config():
    for bin_size in range(5, 25, 5):
        for rand_seed in range(5):
            test_config = ModelConfig(
                num_tanks=3,
                tank_status=[default_tank_state] * 3,
                tank_fill_capacity_kg=200,
                tank_min_capacity_kg=60,
                min_wip_cluster=1,
                max_wip_cluster=4,
                max_wip_magnitude=200,
                wafer_slurry_consumption_kg=3,
                tank_clean_minutes=60,
                tank_fill_minutes=90,
                lookahead_minutes=60 * 8,
                bin_minutes=bin_size,
                rand_seed=rand_seed,
            )
            assert isinstance(test_config.lookahead_period, int)
