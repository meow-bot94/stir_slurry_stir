from typing import List
import numpy as np
import pytest

from src.incoming_wip.incoming_wip_config import IncomingWipConfig
from src.incoming_wip.incoming_wip_generator import IncomingWipGenerator
from src.scenario.scenario_config import default_scenario_config


def generate_config() -> List[IncomingWipConfig]:
    config_list = list()
    for i in range(5):
        incoming_wip_config = IncomingWipConfig(
            scenario=default_scenario_config,
            rand_seed=i*5,
            min_wip_cluster=1,
            max_wip_cluster=4,
            max_wip_magnitude=200,
            wafer_slurry_consumption_kg=3,
        )
        config_list.append(incoming_wip_config)
    return config_list


@pytest.mark.parametrize('config', generate_config())
def test_incoming_wip_simulator(config):
    incoming_wip = IncomingWipGenerator(config).generate()
    assert np.sum(incoming_wip) > 0.1
