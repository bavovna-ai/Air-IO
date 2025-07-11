from .dataset import *
from .dataset_motion import *
from .dataset_utils import *
from .EuRoCdataset import EuRoC
from .BlackBirddataset import BlackBird
from .Pegasusdataset import Pegasus
from .ArduPilotdataset import ArduPilot

DATASET_MAPPING = {
    'euroc': EuRoC,
    'pegasus': Pegasus,
    'blackbird': BlackBird,
    'ardupilot': ArduPilot,
}

def get_dataset(config, data_dir, data_list, mode):
    dataset_class = DATASET_MAPPING.get(config['type'])
    if dataset_class:
        return dataset_class(config, data_dir, data_list, mode)
    else:
        raise ValueError('Unknown dataset type: {}'.format(config['type']))
