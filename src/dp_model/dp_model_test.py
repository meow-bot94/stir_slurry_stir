import pytest

from src.incoming_wip.incoming_wip_generator import IncomingWipGenerator
from src.incoming_wip.incoming_wip_config import default_incoming_wip_config

from src.dp_model.dp_model_config import default_dp_model_config
from src.dp_model.dp_model import DpModel

incoming_wip = IncomingWipGenerator(default_incoming_wip_config).generate()
dp_model = DpModel(default_dp_model_config)
dp_model.run(incoming_wip)
