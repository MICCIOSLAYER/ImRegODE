# basic configuration for the config dict
from typing import Literal
from pathlib import Path
import yaml

from Main_Folder.Modules.utils import deep_update
from Main_Folder.Modules.configuration_setting.logger_configuration import set_logger

standard_log = set_logger(level = 'DEBUG')

def yaml_config_setup(yaml_path: Path,
                      priority: Literal['default', 'personal']='personal') -> dict:
    '''
    use for get and set config values for registrations and also for everithing else in the code/calculations
    for path-like constant use a project_root path as radical and then: image_path_results = Path(project_root/ yaml_dict[image_results])
    Args:
        yaml_path (Path): Path for the file to read, if not existent it will be created in a default_configuration
        priority (Literal['default', 'personal']): if 'personal' the default configuration will be updated with the values in the yaml file, 
            if 'default' the yaml file will be updated with the values in the default configuration
    Returns:
        dict: the dict of the configuration
        Path: the path of the yaml file

    ----------
    USE FOR PATH LIKE CONSTANT
    -----------
    config_dict = yaml_config_setup(PROJECT_PATH / 'config_file.yaml')
    image_res_dir = config_dict['constant']['path-like']['IMG_RESULTS']
    image_results_path = PROJECT_PATH / Path(image_res_dir)
    '''
    

    default_config = {'registrations':{'DRMINE_original':{'n_iterations': 500,
                                                'histo_bins': 64,
                                                'min_delta': float('inf'),
                                                'n_neurons': 100,
                                                'early_stopping_criteria':'',
                                                'crop_style':'',
                                            
                                                'sampling_ratio': 0.1,
                                                'lr':{'MINE':1.e-2, 
                                                    'HomographyNet':{'vL':1.e-3, 
                                                                    'v1':1.e-5
                                                                    },
                                                        }
                                                },  
                                            'elastix_original':{'n_iterations': 5000,
                                                'histo_bins':64,
                                                'n_resolution':4},

                                            'simpleITK_original':{'MMI':{'histo_bins': 100,
                                                    'sampling_ratio': 0.5,
                                                    'lr': 1.e-5,
                                                    'n_iterations': 5000,
                                                    'convergenceMinimumValue': 1.e-9,
                                                    'convergenceWindowSize':200},
                                            'JHMI':{'histo_bins': 100,
                                                    'sampling_ratio': 0.5,
                                                    'lr': 1.e-1,
                                                    'n_iterations': 5000,
                                                    'convergenceMinimumValue': 1.e-9,
                                                    'convergenceWindowSize':200},
                                            'MSE':{'sampling_ratio': 0.5,
                                                    'lr': 1.e-6,
                                                    'n_iterations': 5000,
                                                    'convergenceMinimumValue': 1.e-9,
                                                    'convergenceWindowSize':200},
                                            'NCC':{'sampling_ratio': 0.5,
                                                    'lr': 1.e-1,
                                                    'n_iterations': 5000,
                                                    'convergenceMinimumValue': 1.e-9,
                                                    'convergenceWindowSize':200},
                                                    },
                                            'airlab':{'masked': False,
                                                        'background_values': False,
                                                        'histo_bins':64,
                                                        'metric_sigma':3.0,
                                                        'sampling_ratio':0.1,
                                                        'lr':1.e-4,
                                                        'n_iterations':5000,
                                                        },

                        'Personal':{'TO BE DEFINED': None},
                        },
                        'constant': {'path-like':{'RESULTS': ['Main_Folder', 'Results'],
                                                    'IMG_RESULTS': ['Main_Folder', 'Results', 'ImgRes'],
                                                    'CSV_RESULTS': ['Main_Folder', 'Results', 'CSVResults'],
                                                    'DATA_COLLECTORS': ['Main_Folder', 'Results', 'DataCollectors'],
                                                    'ELASTIX_IMAGE_FOLDER': ['Main_folder', 'Results','ITK-Elastix'],
                                                    'FIRE_DATASET': ['Main_folder', 'Modules', 'dataset', 'FIRE'],
                                                    'CONTROL_POINTS_FOLDER': ['Main_folder', 'Modules', 'dataset', 'FIRE', 'Ground Truth'],
                                                    'IMAGE_MASK': ['Main_folder', 'Modules', 'dataset', 'FIRE', 'Mask'],
                                                    'FIRE_TEST_FOLDER': ['Main_folder', 'Modules', 'dataset', 'FIRE', 'Images', 'Test'],
                                                    'FIRE_REFERENCE_FOLDER': ['Main_folder', 'Modules', 'dataset', 'FIRE', 'Images', 'Reference'],
                                                    'IMAGES': ['Main_folder', 'Modules', 'dataset', 'FIRE', 'Images'],
                                                    'Images_A': ['Main_folder', 'Modules', 'dataset', 'FIRE', 'Images', 'Longitudinal_Studies'],
                                                    'Images_P': ['Main_folder', 'Modules', 'dataset', 'FIRE', 'Images', 'Mosaicing'],
                                                    'Images_S': ['Main_folder', 'Modules', 'dataset', 'FIRE', 'Images', 'Super_Resolution'],},
                                    'text-like':{'TEST_FOLDER': 'Test',
                                                    'REFERENCE_FOLDER': 'Reference',
                                                    'FIXED': '_1',
                                                    'MOVING': '_2',},
                                    'num-like':{'LOWEDGE_BOX': 1.5/9.0,
                                                'HIGHEDGE_BOX': 7.5/9.0,}}
                        }
    if not yaml_path.exists():
        standard_log.info(f'the file {yaml_path.stem} does not exist, a new one is created with default configuration')

        with open(yaml_path, 'w') as f:
            yaml.safe_dump(default_config, f)
        return default_config
    else:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        if priority == 'personal':
            config = deep_update(base_dict=default_config, 
                                 higher_priority_dict=config)

        elif priority == 'default':
            config = deep_update(base_dict=config, 
                                 higher_priority_dict=default_config)
            
        return config