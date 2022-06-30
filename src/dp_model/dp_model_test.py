import pytest

from src.incoming_wip.incoming_wip_generator import IncomingWipGenerator
from src.incoming_wip.incoming_wip_config import default_incoming_wip_config, IncomingWipConfig

from src.dp_model.dp_model_config import default_dp_model_config
from src.dp_model.dp_model import DpModel
from src.scenario.scenario_config import default_scenario_config


def randomized_incoming_wip_config():
    config_list = list()
    for i in range(7):
        incoming_wip_config = IncomingWipConfig(
            scenario=default_scenario_config,
            rand_seed=i,
            min_wip_cluster=1,
            max_wip_cluster=4,
            max_wip_magnitude=200,
            wafer_slurry_consumption_kg=3,
        )
        config_list.append(incoming_wip_config)
    return config_list


@pytest.mark.parametrize('config', randomized_incoming_wip_config())
def test_dp_model(config):
    incoming_wip = IncomingWipGenerator(config).generate()
    dp_model = DpModel(default_dp_model_config)
    dp_model.run(incoming_wip)
    state_df = dp_model.get_best_state_df()
    best_end_state = dp_model.get_best_end_state()
    print(state_df)
    assert best_end_state[1].cost == 0

