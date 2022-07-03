from collections import namedtuple

State = namedtuple('State', ['period', 'active_tank', 'tank_states'])
StateValue = namedtuple('StateValue', ['cost', 'origin'])
