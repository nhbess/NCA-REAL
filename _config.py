# TRAINING PARAMETERS
DEFAULT_PARAMETERS = {
        'LEARNING_RATE' :   2e-3,
        'WEIGHT_DECAY' :    0,
        'MILESTONES' :      [3000, 6000, 8000],
        'GAMMA' :           1,
        'BOARD_SHAPE' :     [3,3],
        'BATCH_SIZE' :      8,
        'POOL_SIZE' :       1000,
        'TRAINING_STEPS' :  10,
        'UPDATE_STEPS' :    [15, 30],
        'NUM_MOVEMENTS' :   10,
        'NEIGHBORHOOD' :    'Chebyshev',
        'DEAD_PERCENTAGE' : [0,25], # %
    }

LEARNING_RATE =     None
WEIGHT_DECAY =      None
MILESTONES =        None
GAMMA =             None
BOARD_SHAPE =       None
BATCH_SIZE =        None
POOL_SIZE =         None
TRAINING_STEPS =    None
UPDATE_STEPS =      None
NUM_MOVEMENTS =     None
NEIGHBORHOOD =      None
DEAD_PERCENTAGE =   None

def training_parameters() -> dict:
    data = {}
    for key in DEFAULT_PARAMETERS.keys():
        data[key] = globals()[key]
    return data

def set_parameters(configuration: dict = None):
    for key, value in DEFAULT_PARAMETERS.items():
        globals()[key] = value

    if configuration is not None:
        for key, value in configuration.items():
            if key not in DEFAULT_PARAMETERS:
                raise Exception(f'DEFAULT_PARAMETERS does not have the attribute {key}')
            globals()[key] = value

set_parameters()