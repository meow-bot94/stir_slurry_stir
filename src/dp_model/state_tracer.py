from typing import Tuple, Dict, List, TYPE_CHECKING

import pandas as pd

from src.dp_model.exceptions import ModelNotRunError
from src.dp_model.model_states import State, StateValue


if TYPE_CHECKING:
    from src.dp_model.dp_model import DpModel


class StateTracer:
    def __init__(self, dp_model: 'DpModel'):
        self.model = dp_model
        self.states = dp_model.states

    def _check_model_has_run(self) -> bool:
        if not self.model.has_run:
            raise ModelNotRunError(f'Mode not run: run model before querying end states')
        return True

    def get_best_end_state(self) -> Tuple[State, StateValue]:
        self._check_model_has_run()
        end_states: Dict[State, StateValue] = self.states[self.model.lookahead_period]
        # Bug in Pycharm's type hinting: https://youtrack.jetbrains.com/issue/PY-38897
        # noinspection PyTypeChecker
        best_state, best_state_value = min(end_states.items(), key=self.model.objective_function)
        return best_state, best_state_value

    def trace_best_states(self) -> List[Tuple[State, StateValue]]:
        best_state, best_state_value = self.get_best_end_state()
        reverse_list = list()
        reverse_list.append((best_state, best_state_value))
        current_state = best_state
        current_value = best_state_value
        while current_state.period > 0:
            next_state = current_value.origin
            next_value = self.states[current_state.period-1][next_state]
            reverse_list.append((next_state, next_value))
            current_state, current_value = next_state, next_value
        state_list = reversed(reverse_list)
        return list(state_list)

    def get_best_state_df(self) -> pd.DataFrame:
        best_actions = self.trace_best_states()
        state_list = list()
        for state, state_value in best_actions:
            state_dict = dict()
            state_dict['period'] = state.period
            state_dict['active_tank'] = state.active_tank
            for tank_state in state.tank_states:
                tank_id = tank_state.id
                state_dict[f'current_slurry_unit_{tank_id}'] = tank_state.current_slurry_unit
                state_dict[f'cleaning_period_remaining_{tank_id}'] = tank_state.cleaning_period_remaining
                state_dict[f'filling_period_remaining_{tank_id}'] = tank_state.filling_period_remaining
                state_dict[f'down_start_period_{tank_id}'] = tank_state.down_start_period
            state_dict['cost'] = state_value.cost
            state_list.append(state_dict)
        state_df = pd.DataFrame(state_list)
        return state_df
