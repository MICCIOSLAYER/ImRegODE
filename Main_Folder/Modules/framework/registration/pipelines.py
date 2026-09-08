# Creation of wrapper pipeline as default to use them as singular function, 

import sys, os
import torch
from torch.utils.data import Dataset, DataLoader
from itertools import islice
from tqdm.auto import tqdm
from pathlib import Path


from typing import Callable, Optional, Tuple, Sequence
from Main_Folder.Modules.framework.data_classes  import Registration_Data_Collector
from Main_Folder.Modules.configuration_setting.yaml_configuration  import FrameworkConfig
from Main_Folder.Modules.framework.data_classes  import SampleDict

def run_registration_pipeline(
        dataset : Dataset, #optional
        dataloader : DataLoader, # depending on the first
        config_dict : dict | FrameworkConfig,
        preprocessing_fn : Callable[[SampleDict, dict], SampleDict],
        registratoion_fn : Callable, # to typize
        data_collection_fn : Callable[[dict], tuple[Registration_Data_Collector, Path |None]],
        save_results: bool = True
        )->Tuple[Registration_Data_Collector, Optional[Path]]:
    '''
    _summary_

    Args:
        dataset (Dataset): dataset of already organized data
        dataloader (DataLoader): dataloader starting from the dataset, previously cited
        config_dict (dict or FrameworkConfig): dict of configuration
        preprocessing_fn (Callable[[SampleDict, dict], SampleDict]): preprocessing function, all from dataset/dataloader to input of registration_fn
        registratoion_fn (Callable): registration/wrappper to get the results in dict form
        data_collection_fn (Callable [dict, [Path, Registration_Data_Collector]]): the function to get results organized in Registration_Data_Collector

    Returns:
        [Registration_Data_Collector, Optional[Path]]: Registration_Data_Collector is a must, if save_results, also the saving_path is delivered
    '''

    data_collection = Registration_Data_Collector()
    return data_collection