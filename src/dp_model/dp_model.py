from abc import ABC, abstractmethod

from src.dp_model.exceptions import ModelNotRunError
from src.dp_model.model_states import State, StateValue
from src.dp_model.state_tracer import StateTracer
from src.dp_model.tank_state import TankState
from typing import List, Tuple, Dict, Type, Callable

from src.dp_model.dp_model_config import DPModelConfig
import numpy as np
import pandas as pd
from src.dp_model.initial_dp_tank_state import InitialDPTankState
import operator

from src.scenario.scenario_config import ScenarioConfig


class DpModel:
    origin_state = State(None, None, None)

    def __init__(self, config: DPModelConfig):
        self.config: DPModelConfig = config
        self.scenario: ScenarioConfig = self.config.scenario
        self.tank_config: Dict[int, InitialDPTankState] = self._init_tank_config()
        self.tank_count: int = len(self.tank_config)
        self.lookahead_period: int = self.scenario.lookahead_period
        self.states: Dict[int, Dict[State, StateValue]] = self._init_dp_states()
        self.has_run = False

    def _init_tank_config(self) -> Dict[int, InitialDPTankState]:
        return {idx: config for idx, config in enumerate(self.config.initial_tank_status)}

    def _get_initial_dp_state(self):
        tank_states = tuple(self.config.init_tank_states())
        initial_state = State(period=0, active_tank=self.config.active_tank_index, tank_states=tank_states)
        return initial_state

    def _init_dp_states(self):
        states = {period: dict() for period in range(self.lookahead_period+1)}
        states[0][self._get_initial_dp_state()] = StateValue(0, self.origin_state)
        return states

    def _discretize_incoming_wip(self, incoming_wip: np.array) -> np.array:
        discrete_incoming_wip = np.round(incoming_wip / self.scenario.slurry_unit_bin_kg).astype('int')
        return discrete_incoming_wip

    def get_next_tank(self, tank_idx: int) -> int:
        return (tank_idx + 1) % self.tank_count

    def _get_other_tanks_states(self, current_state: State, exclude_tank_list: List[TankState]) -> List[TankState]:
        exclude_idx = set([tank.id for tank in exclude_tank_list])
        other_tank_idx = [idx for idx in self.tank_config if idx not in exclude_idx]
        other_tank_states = [current_state.tank_states[idx] for idx in other_tank_idx]
        return other_tank_states

    def check_tank_useable(self, tank_state: TankState) -> bool:
        return tank_state.down_start_period == np.inf

    def _process_cleaning_tank(self, tank_state: TankState) -> TankState:
        return TankState(tank_state.id, tank_state.current_slurry_unit, tank_state.cleaning_period_remaining-1, tank_state.filling_period_remaining, tank_state.down_start_period)

    def _fill_selected_tank(self, tank_state: TankState) -> TankState:
        after_fill_period_remaining = tank_state.filling_period_remaining - 1
        if after_fill_period_remaining == 0:
            tank_config = self.tank_config[tank_state.id]
            max_slurry_unit = tank_config.tank_fill_capacity_slurry_unit
            return TankState(tank_state.id, max_slurry_unit, tank_state.cleaning_period_remaining, after_fill_period_remaining, np.inf)
        else:
            return TankState(tank_state.id, tank_state.current_slurry_unit, tank_state.cleaning_period_remaining, after_fill_period_remaining, tank_state.down_start_period)

    def _fill_tank_if_possible(self, tank_states: List[TankState]) -> List[TankState]:
        if len(tank_states) == 0:
            return []
        earliest_tank_state = min(tank_states, key=operator.attrgetter('down_start_period'))
        fill_tank = self._fill_selected_tank(earliest_tank_state)
        other_tanks = [tank_state for tank_state in tank_states if tank_state.id != fill_tank.id]
        other_tanks.append(fill_tank)
        return other_tanks

    def _process_non_active_tanks(self, tank_states: List[TankState]) -> List[TankState]:
        idle_full_tanks = list(filter(lambda state: state.down_start_period == np.inf, tank_states))
        cleaning_tanks = list(filter(lambda state: state.cleaning_period_remaining > 0, tank_states))
        filling_tanks = list(filter(lambda state: (state.cleaning_period_remaining == 0) and (state.filling_period_remaining > 0), tank_states))

        new_cleaning_tanks = list(map(self._process_cleaning_tank, cleaning_tanks))
        new_filling_tanks = self._fill_tank_if_possible(filling_tanks)
        return new_filling_tanks + new_cleaning_tanks + idle_full_tanks

    def _put_tank_to_clean(self, tank_state: TankState, period: int) -> TankState:
        tank_id = tank_state.id
        tank_config = self.tank_config[tank_id]
        return TankState(tank_id, 0, tank_config.tank_clean_period, tank_config.tank_fill_period, period)

    def _add_slurry(self, tank_state: TankState, slurry_unit: int) -> TankState:
        return TankState(
            tank_state.id,
            tank_state.current_slurry_unit + slurry_unit,
            tank_state.cleaning_period_remaining,
            tank_state.filling_period_remaining,
            tank_state.down_start_period,
        )

    def _consume_slurry(self, tank_state: TankState, slurry_unit: int) -> TankState:
        return TankState(
            tank_state.id,
            tank_state.current_slurry_unit - slurry_unit,
            tank_state.cleaning_period_remaining,
            tank_state.filling_period_remaining,
            tank_state.down_start_period,
        )

    def _get_next_period(self, current_state: State) -> int:
        return current_state.period + 1

    def _do_nothing(self, current_state: State, cost: float, slurry_consumption: int) -> Tuple[State, float]:
        next_period = self._get_next_period(current_state)
        active_tank_state = current_state.tank_states[current_state.active_tank]
        active_tank_current_config = self.tank_config[current_state.active_tank]
        active_tank_current_slurry_unit = active_tank_state.current_slurry_unit

        new_tank_states = list()
        new_slurry_unit = active_tank_current_slurry_unit - slurry_consumption
        if new_slurry_unit <= active_tank_current_config.tank_min_capacity_slurry_unit:  # hit min capacity
            next_tank_idx = self.get_next_tank(current_state.active_tank)
            next_tank_state = current_state.tank_states[next_tank_idx]
            if self.check_tank_useable(next_tank_state):  # attempt to switch
                next_tank_new_state = self._add_slurry(next_tank_state, new_slurry_unit)
                active_tank_new_state = self._put_tank_to_clean(active_tank_state, current_state.period)
                new_tank_states.append(next_tank_new_state)
                new_tank_states.append(active_tank_new_state)
                other_tank_states = self._get_other_tanks_states(current_state, new_tank_states)
                processed_other_tank_states = self._process_non_active_tanks(other_tank_states)
                new_tank_states.extend(processed_other_tank_states)
                new_state = State(next_period, next_tank_idx, tuple(sorted(new_tank_states)))
                return new_state, cost
        # No where to switch, or no need to switch
        unserved_slurry = max(slurry_consumption - active_tank_current_slurry_unit, 0)
        served_slurry = slurry_consumption - unserved_slurry
        new_cost = cost + unserved_slurry * self.config.slurry_unit_violate_penalty
        active_tank_new_state = self._consume_slurry(active_tank_state, served_slurry)
        new_tank_states.append(active_tank_new_state)
        other_tank_states = self._get_other_tanks_states(current_state, new_tank_states)
        processed_other_tank_states = self._process_non_active_tanks(other_tank_states)
        new_tank_states.extend(processed_other_tank_states)

        new_active_tank = current_state.active_tank
        new_state = State(next_period, new_active_tank, tuple(sorted(new_tank_states)))
        return new_state, new_cost

    def _switch_out_active_tank(self, current_state: State, cost: float, slurry_consumption: int) -> Tuple[State, float]:
        new_tank_states = list()
        new_active_tank = self.get_next_tank(current_state.active_tank)
        next_tank_state = current_state.tank_states[new_active_tank]
        if not self.check_tank_useable(next_tank_state):
            return self.origin_state, np.inf
        active_tank_state = current_state.tank_states[current_state.active_tank]
        active_tank_current_config = self.tank_config[current_state.active_tank]
        active_tank_current_slurry_unit = active_tank_state.current_slurry_unit
        active_tank_min_slurry = active_tank_current_config.tank_min_capacity_slurry_unit

        wasted_slurry_units = max(active_tank_current_slurry_unit - active_tank_min_slurry, 0)
        new_cost = cost + wasted_slurry_units * self.config.slurry_unit_wastage_penalty

        bringover_slurry_units = min(active_tank_min_slurry, active_tank_current_slurry_unit)
        next_tank_new_state = self._add_slurry(next_tank_state, bringover_slurry_units-slurry_consumption)
        active_tank_new_state = self._put_tank_to_clean(active_tank_state, current_state.period)
        new_tank_states.append(next_tank_new_state)
        new_tank_states.append(active_tank_new_state)
        other_tank_states = self._get_other_tanks_states(current_state, new_tank_states)
        processed_other_tank_states = self._process_non_active_tanks(other_tank_states)
        new_tank_states.extend(processed_other_tank_states)
        next_period = self._get_next_period(current_state)
        new_state = State(next_period, new_active_tank, tuple(sorted(new_tank_states)))
        return new_state, new_cost

    @property
    def available_actions(self) -> List[Callable]:
        return [
            self._do_nothing,
            self._switch_out_active_tank
        ]

    def _add_state(self, current_state: State, next_state: State, next_cost: float):
        if next_state.period is None:
            return
        states_in_period = self.states[next_state.period]
        old_state_value = states_in_period.get(next_state, StateValue(np.inf, self.origin_state))
        if next_cost < old_state_value.cost:
            states_in_period[next_state] = StateValue(next_cost, current_state)

    def run(self, incoming_wip: np.array):
        if self.has_run is True:
            return self
        discretized_wip = self._discretize_incoming_wip(incoming_wip)
        for period in range(self.lookahead_period):
            for current_state, current_state_value in self.states[period].items():
                cost: float = current_state_value.cost
                slurry_consumption: int = discretized_wip[period]
                for action in self.available_actions:
                    new_state, new_cost = action(current_state, cost, slurry_consumption)
                    self._add_state(current_state, new_state, new_cost)
        self.has_run = True
        return self

    @staticmethod
    def objective_function(state_tuple: Tuple[State, StateValue]) -> float:
        state, state_value = state_tuple
        total_slurry = sum(tank.current_slurry_unit for tank in state.tank_states)
        return state_value.cost - total_slurry

    def get_best_end_state(self) -> Tuple[State, StateValue]:
        if not self.has_run:
            raise ModelNotRunError(f'Mode not run: run model before querying end states')
        return StateTracer(self).get_best_end_state()

    def trace_best_states(self) -> List[Tuple[State, StateValue]]:
        return StateTracer(self).trace_best_states()

    def get_best_state_df(self) -> pd.DataFrame:
        return StateTracer(self).get_best_state_df()


# class ModelAction(ABC):
#     def __init__(self, dp_model: DpModel, current_state: State, cost: float, slurry_consumption: int):
#         self.model = dp_model
#         self.current_state = current_state
#         self.cost = cost
#         self.slurry_consumption = slurry_consumption
#
#     @abstractmethod
#     def is_feasible(self) -> bool:
#         pass
#
#     def _get_next_period(self) -> int:
#         return self.current_state.period + 1
#
#     @abstractmethod
#     def _do(self, current_state: State, cost: float, slurry_consumption: int) -> Tuple[List[TankState], int, float]:
#         pass
#
#     def do(self) -> Tuple[State, float]:
#         new_tank_states, new_active_tank, new_cost = self._do(self.current_state, self.cost, self.slurry_consumption)
#         next_period = self._get_next_period()
#         new_state = State(next_period, new_active_tank, tuple(sorted(new_tank_states)))
#         return new_state, new_cost
#
#
# class DoNothing(ModelAction):
#     def is_feasible(self) -> bool:
#         next_tank_idx = self.model.get_next_tank(self.current_state.active_tank)
#         next_tank_state = self.current_state.tank_states[next_tank_idx]
#         return self.model.check_tank_useable(next_tank_state)
#
#     def _do(self, current_state: State, cost: float, slurry_consumption: int) -> Tuple[List[TankState], int, float]:
#         model = self.model
#         active_tank_state = current_state.tank_states[current_state.active_tank]
#         active_tank_current_config = model.tank_config[current_state.active_tank]
#         active_tank_current_slurry_unit = active_tank_state.current_slurry_unit
#
#         new_tank_states = list()
#         new_slurry_unit = active_tank_current_slurry_unit - slurry_consumption
#         if new_slurry_unit <= active_tank_current_config.tank_min_capacity_slurry_unit:  # hit min capacity
#             next_tank_idx = model.get_next_tank(current_state.active_tank)
#             next_tank_state = current_state.tank_states[next_tank_idx]
#             if model.check_tank_useable(next_tank_state):  # attempt to switch
#                 next_tank_new_state = model._add_slurry(next_tank_state, new_slurry_unit)
#                 active_tank_new_state = model._put_tank_to_clean(active_tank_state, current_state.period)
#                 new_tank_states.append(next_tank_new_state)
#                 new_tank_states.append(active_tank_new_state)
#                 other_tank_states = model._get_other_tanks_states(current_state, new_tank_states)
#                 processed_other_tank_states = model._process_non_active_tanks(other_tank_states)
#                 new_tank_states.extend(processed_other_tank_states)
#                 return new_tank_states, next_tank_idx, cost
#         # No where to switch, or no need to switch
#         unserved_slurry = max(slurry_consumption - active_tank_current_slurry_unit, 0)
#         served_slurry = slurry_consumption - unserved_slurry
#         new_cost = cost + unserved_slurry * model.config.slurry_unit_violate_penalty
#         active_tank_new_state = model._consume_slurry(active_tank_state, served_slurry)
#         new_tank_states.append(active_tank_new_state)
#         other_tank_states = model._get_other_tanks_states(current_state, new_tank_states)
#         processed_other_tank_states = model._process_non_active_tanks(other_tank_states)
#         new_tank_states.extend(processed_other_tank_states)
#         return new_tank_states, current_state.active_tank, new_cost
#
#
# class SwitchOutActiveTank(ModelAction):
#     def is_feasible(self) -> bool:
#         return True
#
#     def _do(self, current_state: State, cost: float, slurry_consumption: int) -> Tuple[List[TankState], int, float]:
#         model = self.model
#         new_tank_states = list()
#         next_tank_idx = model.get_next_tank(current_state.active_tank)
#         next_tank_state = current_state.tank_states[next_tank_idx]
#
#         active_tank_state = current_state.tank_states[current_state.active_tank]
#         active_tank_current_config = model.tank_config[current_state.active_tank]
#         active_tank_current_slurry_unit = active_tank_state.current_slurry_unit
#         active_tank_min_slurry = active_tank_current_config.tank_min_capacity_slurry_unit
#
#         wasted_slurry_units = max(active_tank_current_slurry_unit - active_tank_min_slurry, 0)
#         new_cost = cost + wasted_slurry_units * model.config.slurry_unit_wastage_penalty
#
#         bringover_slurry_units = min(active_tank_min_slurry, active_tank_current_slurry_unit)
#         next_tank_new_state = model._add_slurry(next_tank_state, bringover_slurry_units-slurry_consumption)
#         active_tank_new_state = model._put_tank_to_clean(active_tank_state, current_state.period)
#         new_tank_states.append(next_tank_new_state)
#         new_tank_states.append(active_tank_new_state)
#         other_tank_states = model._get_other_tanks_states(current_state, new_tank_states)
#         processed_other_tank_states = model._process_non_active_tanks(other_tank_states)
#         new_tank_states.extend(processed_other_tank_states)
#         return new_tank_states, next_tank_idx, new_cost


