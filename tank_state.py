from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass, field
from typing import List
from functools import total_ordering


@total_ordering
@dataclass
class TankState:
    cleaning_period_remaining: int
    filling_period_remaining: int
    current_slurry_kg: float

    cmp_score: float = field(init=False)

    def __post_init__(self):
        self.cmp_score = self.current_slurry_kg - self.filling_period_remaining - self.cleaning_period_remaining
        if self.current_slurry_kg > 0:
            assert self.filling_period_remaining == 0
            assert self.cleaning_period_remaining == 0
        if self.cleaning_period_remaining > 0:
            assert self.filling_period_remaining > 0

    def __eq__(self, other: TankState):
        return self.cmp_score < other.cmp_score

    def __ne__(self, other: TankState):
        return not (self == other)

    def __lt__(self, other: TankState):
        return self.cmp_score < other.cmp_score


default_tank_state = TankState(
    cleaning_period_remaining=0, filling_period_remaining=0, current_slurry_kg=200,
)
