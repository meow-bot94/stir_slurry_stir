from collections import namedtuple

TankState = namedtuple(
    'TankState',
    ['id', 'current_slurry_unit', 'cleaning_period_remaining', 'filling_period_remaining', 'down_start_period']
)
