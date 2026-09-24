# basic configuration for the config dict
from typing import Literal
from pathlib import Path
import yaml
import sys

from Main_Folder.Modules.utils import get_root_path, concatenate_paths, deep_update
from Main_Folder.Modules.configuration_setting.logger_configuration import get_logger

root_path = get_root_path()
standard_log = get_logger(__name__)

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
    

    default_config = {'registrations':{'DRMINE_original':{'n_iterations': 0,
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
        standard_log.warning(f'the file {yaml_path.stem} does not exist, a new one is created with default configuration')

        with open(yaml_path, 'w') as f:
            yaml.safe_dump(default_config, f)
        return default_config
    else:
        standard_log.debug(f' opening the file {yaml_path.stem} to downloading the dict')
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        if priority == 'personal':
            config = deep_update(base_dict=default_config, 
                                 higher_priority_dict=config)

        elif priority == 'default':
            config = deep_update(base_dict=config, 
                                 higher_priority_dict=default_config)
        
        # ------------------
        # RESOLVE PATHS VALS
        # ------------------
        
        path_config_dict = config['constant']['path-like']
        standard_log.debug(f'prior conversion of path-like: the RESULTS term is {path_config_dict['RESULTS']}')
        path_dict = {k: concatenate_paths(root= root_path, relative_path= Path(*path_config_dict[k])) for k in path_config_dict.keys()}
        for k, v in path_dict.items():
            config['constant']['path-like'][k] = v
        standard_log.debug(f'after conversion of paths-like, the RESULTS term began:{config['constant']['path-like']['RESULTS']}  ')
        return config


yaml_file_path = concatenate_paths(root = root_path, relative_path='config_file.yaml')

class FrameworkConfig:
    ' a class to use the config_dict in modules'
    def __init__(self,
                    yaml_path: Path= yaml_file_path,
                    priority : Literal['personal', 'default'] = 'personal'):
        self.yaml_path = yaml_path
        self.priority = priority
        self._config_dict = yaml_config_setup(yaml_path=yaml_path, priority=priority)
        self.colab_execution = 'google.colab' in sys.modules

        
    def reload_yaml(self):
        new_config=yaml_config_setup(yaml_path=self.yaml_path, 
                                     priority='personal')
        self._config_dict.clear()
        return self._config_dict.update(new_config)

    def reset_default(self):
        new_config=yaml_config_setup(yaml_path=self.yaml_path, 
                                        priority='default')
        self._config_dict.clear()
        return self._config_dict.update(new_config)

    @property
    def config_dict(self):
        return self._config_dict
    

    @property
    def path_dict(self):
        if self.colab_execution:
            colab_path_dict = self._config_dict['constant']['path-like'].copy()

            colab_path_dict['FIRE_DATASET'] = colab_path_dict['DRIVE_FIRE']
            colab_path_dict['CONTROL_POINTS_FOLDER'] = colab_path_dict['FIRE_DATASET'] / 'Groung Truth'
            colab_path_dict['IMAGES'] = colab_path_dict['FIRE_DATASET'] / 'Images'
            colab_path_dict['IMAGE_MASK'] = colab_path_dict['FIRE_DATASET'] / 'Mask'
            colab_path_dict['FIRE_REFERENCE_FOLDER'] = colab_path_dict['IMAGES'] / 'Reference'
            colab_path_dict['FIRE_TEST_FOLDER'] = colab_path_dict['IMAGES'] / 'Test'
            # NOTE no need to convert Image_A/S/P since no


            colab_path_dict['RESULTS'] = colab_path_dict['DRIVE_RESULTS']
            colab_path_dict['IMG_RESULTS'] = colab_path_dict['RESULTS'] / 'ImgRes'
            colab_path_dict['DATA_COLLECTORS'] = colab_path_dict['RESULTS'] / 'DataCollectors'
            colab_path_dict['CSV_RESULTS'] =colab_path_dict['RESULTS'] / 'CSVResults'
            colab_path_dict['ELASTIX_IMAGE_FOLDER'] = colab_path_dict['RESULTS'] / 'ITK-Elastix'
            
            


        return self._config_dict['constant']['path-like']


    @property
    def num_dict(self):
        return self._config_dict['constant']['num-like']


    @property
    def text_dict(self):
        return self._config_dict['constant']['text-like']

    
    @property
    def registrations(self):
        return self._config_dict['registrations']



    