from .dataset import *
from .dataset_motion import *
from .dataset_utils import *
from .EuRoCdataset import EuRoCdataset
from .BlackBirddataset import BlackBirdDataset
from .Pegasusdataset import PegasusDataset
from .ArduPilotdataset import ArduPilotDataset

DATASET_MAPPING = {
    'euroc': EuRoCdataset,
    'pegasus': PegasusDataset,
    'blackbird': BlackBirdDataset,
    'ardupilot': ArduPilotDataset,
}

def get_dataset(config, data_dir, data_list, mode):
    dataset_class = DATASET_MAPPING.get(config['type'])
    if dataset_class:
        return dataset_class(config, data_dir, data_list, mode)
    else:
        raise ValueError('Unknown dataset type: {}'.format(config['type']))
