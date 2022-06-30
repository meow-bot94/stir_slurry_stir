from dataclasses import dataclass, field

@dataclass
class ScenarioConfig:
    lookahead_minutes: int
    period_bin_minutes: int  # How many minutes is one unit of time
    slurry_unit_bin_kg: float  # How many kg is one unit of slurry

    lookahead_period: int = field(init=False)

    def __post_init__(self):
        self.lookahead_period = int(self.lookahead_minutes / self.period_bin_minutes)


default_scenario_config = ScenarioConfig(
    lookahead_minutes=60 * 8,
    period_bin_minutes=10,
    slurry_unit_bin_kg=1,
)
