import numpy as np
import random
from scipy.stats import gamma

from model_config import ModelConfig


class IncomingWipGenerator:

    def __init__(self, config: ModelConfig):
        self.config = config
        random.seed(config.rand_seed)
        np.random.seed(config.rand_seed)

    def _init_lookahead_wip(self):
        return np.zeros(self.config.lookahead_period)

    def _generate_wafer_cluster_count(self) -> int:
        return random.randrange(self.config.min_wip_cluster, self.config.max_wip_cluster)

    def _generate_wip(self) -> np.array:

        lookahead_array = list(range(self.config.lookahead_period))
        wip_arrival_time = np.random.uniform(0, self.config.lookahead_period)
        scale = 0.1 * self.config.lookahead_period
        wip = gamma.pdf(x=lookahead_array, a=3, scale=scale, loc=wip_arrival_time)

        return wip

    def _generate_expected_wip_consumption(self) -> np.array:
        wip = self._generate_wip()
        wip_magnitude = np.random.uniform(self.config.max_wip_magnitude)
        wip_consumption = wip * wip_magnitude * self.config.wafer_slurry_consumption_kg
        return wip_consumption

    def generate(self):
        lookahead_wip = self._init_lookahead_wip()
        wip_cluster_count = self._generate_wafer_cluster_count()
        for i in range(wip_cluster_count):
            lookahead_wip += self._generate_expected_wip_consumption()
        return lookahead_wip

