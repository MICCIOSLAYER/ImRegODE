# for preprocessing of various registration method

from typing import Callable, TypedDict, Optional, Tuple, Sequence, Required, NotRequired, Union

from Main_Folder.Modules.framework.data_classes import SampleDict
from Main_Folder.Modules.framework.visualization import _image_to_numpy
from Main_Folder.Modules.configuration_setting.logger_configuration import get_logger
from Main_Folder.Modules.configuration_setting.yaml_configuration import FrameworkConfig
import torch
import SimpleITK as sitk
import itk
from itertools import islice
from airlab.utils.image import Image as AirlabImage # NOTE consider to move airlab from TESI_MAGISTRALE to another folder
from pathlib import Path
import logging
from fractions import Fraction
image_type = Union[torch.Tensor, itk.Image, sitk.Image, AirlabImage]
#---------------------
#REGISTRATION PIPELINE
#---------------------
# ALREADY DEFINED: 
# DATASET ---> DataLoader to be inserted in the pipeline


standard_log = get_logger(__name__)
_default_framework_cfg = FrameworkConfig().config_dict


#------------
#LIST OF PREPROCESSING
#-----------------
def general_preprocessing(sample_dict: Union[SampleDict, Sequence],
                       config_dict: dict=_default_framework_cfg,
                       transformations : Optional[Tuple[Callable[[SampleDict, dict], SampleDict], ...]] = None,
                       )-> SampleDict:
    '''
    The point is to adapt the most of preprocessing function to be inserted in an ordinated list to use them in an ordered and methodical way during the pipeline
    For simplicity all of them must have the same input data type as for the same output data type

    ----
    IMPORTANT
    ----------
    if the sample_dict is the item from dataset.__getitem__(index) then the elements must be sorted out as:
            sample_dict = ('reference_sample', 'test_sample', 'reference_sample_path', 'test_sample_path').
    
    ----------- 
    Parameters
    ----------- 

        sample_dict (SampleDict): Starting point, constructed by the dataset 
        config_dict (dict): possibly taking by a yaml file to use the method dict_type.get('key', default) to avoid errors
        transformations (Tuple[Callable, ...]): An ordinated tuple of preprocessing functions to be applied in order, they have to have the same input and output type to be used in this way.

    Returns
    ---------

        dict: the dict has to be almost on the same type of SampleDict to better flexibility during pipeline
    '''
    preprocessed_sample = sample_dict
    
    if isinstance(sample_dict, Sequence) and not isinstance(sample_dict, dict):
        
        test_image, reference_image, test_path, reference_path = sample_dict
        if isinstance(test_path, (tuple, list)):
            test_path=test_path[0]
        if isinstance(reference_path, (tuple, list)):
            reference_path = reference_path[0] 
        standard_log.info(f'preprocessing sample shape: {test_image.shape}, {reference_image.shape}')
        preprocessed_sample = {
            'reference_sample': reference_image,
            'test_sample': test_image,
            'reference_sample_path': Path(reference_path),
            'test_sample_path': Path(test_path),
            'sample_name': Path(reference_path).stem.split('_')[0],
        }
    if transformations:
        for transformation in transformations:
            preprocessed_sample = transformation(preprocessed_sample, config_dict)
    else: 
        standard_log.info('since no trasformation required and the data input is the output of a dataset, the transformation will consist in sorting elements')
    standard_log.info(f'reference/test image sample shape: {preprocessed_sample["reference_sample"].shape}')
    return preprocessed_sample



def sitk_preprocessing(sample_dict : SampleDict, 
                       config_dict: dict = _default_framework_cfg
                       )-> dict:
    '''
    A preprocessing function to adjust the images for the registration pipeline using SimpleITK. It takes a sample dictionary and a configuration dictionary as input and returns a preprocessed sample dictionary.
    For this experiment: 
    MMI-sitk: INPUT - np.ndarray | torch.Tensor | Path, OUTPUT - tuple[outTx{Any}, sitk.ImageRegistrationMethod, float]
    JHMI-sitk: INPUT - np.ndarray | torch.Tensor | Path, OUTPUT - tuple[outTx{Any}, sitk.ImageRegistrationMethod, float]
    MSE-sitk: INPUT - np.ndarray | torch.Tensor | Path, OUTPUT - tuple[outTx{Any}, sitk.ImageRegistrationMethod, float]
    NCC-sitk: INPUT - np.ndarray | torch.Tensor | Path, OUTPUT - tuple[outTx{Any}, sitk.ImageRegistrationMethod, float]
    Args:
        sample_dict (SampleDict): A dictionary containing the reference and test samples, their paths, and the sample name.
        config_dict (dict): A dictionary containing configuration parameters for preprocessing.

    Returns:
        SampleDict: A dictionary containing the preprocessed reference and test samples.
    '''
    transformations = None
    preprocessed_sample = general_preprocessing(sample_dict = sample_dict,
                                                config_dict=config_dict,
                                                transformations=transformations)
    return preprocessed_sample



def airlab_preprocessing(sample_dict : SampleDict, 
                       config_dict: dict = _default_framework_cfg
                       )-> dict:
    '''
    A preprocessing function to adjust the images for the registration pipeline using Airlab. It takes a sample dictionary and a configuration dictionary as input and returns a preprocessed sample dictionary.
    AMI: INPUT - np.ndarray | torch.Tensor | Path, OUTPUT - tuple[torch.Tensor, dict]
    in the middle from input to 
    Args:
        sample_dict (SampleDict): A dictionary containing the reference and test samples, their paths, and the sample name.
        config_dict (dict): A dictionary containing configuration parameters for preprocessing.

    Returns:
        SampleDict: A dictionary containing the preprocessed reference and test samples.
    '''
    #===========READ IMAGES========
    transformations = None
    preprocessed_sample = general_preprocessing(sample_dict = sample_dict,
                                                config_dict=config_dict,
                                                transformations=transformations)
    

    return preprocessed_sample



def drmine_preprocessing(sample_dict : SampleDict, 
                       config_dict: dict = _default_framework_cfg
                       )-> dict:
    '''
    A preprocessing function to adjust the images for the registration pipeline using DeepReg. It takes a sample dictionary and a configuration dictionary as input and returns a preprocessed sample dictionary.

    Args:
        sample_dict (SampleDict): A dictionary containing the reference and test samples, their paths, and the sample name.
        config_dict (dict): A dictionary containing configuration parameters for preprocessing.
    Returns:
        SampleDict: A dictionary containing the preprocessed reference and test samples.
    '''

    transformations = None
    preprocessed_sample = general_preprocessing(sample_dict = sample_dict,
                                                config_dict=config_dict,
                                                transformations=transformations)
    return preprocessed_sample



def elastix_preprocessing(sample_dict : SampleDict, 
                       config_dict: dict = _default_framework_cfg
                       )-> dict:
    '''
    A preprocessing function to adjust the images for the registration pipeline using Elastix. It takes a sample dictionary and a configuration dictionary as input and returns a preprocessed sample dictionary.

    Args:
        sample_dict (SampleDict): A dictionary containing the reference and test samples, their paths, and the sample name.
        config_dict (dict): A dictionary containing configuration parameters for preprocessing.
    Returns:
        SampleDict: A dictionary containing the preprocessed reference and test samples.
    '''
    transformations = None
    preprocessed_sample = general_preprocessing(sample_dict = sample_dict,
                                                config_dict=config_dict,
                                                transformations=transformations)
    return preprocessed_sample


#===================
# CROP FNS
#==================
def crop_images(moving_image : Path | torch.Tensor,
                fixed_image : Path | torch.Tensor,
                config_dict : dict | FrameworkConfig,
                show_images : bool = False,
                )-> Tuple[torch.Tensor, torch.Tensor]:
    '''
    given the images it crop to the desired size 
    if requested the images are shown as confront between the original and the cropped
    
    Parameters:
    moving_image : Path | torch.Tensor
        the image to crop, it can be a path or a tensor given by the dataloader
    fixed_image : Path | torch.Tensor
        the image to crop, it can be a path or a tensor given by the dataloader 
    show_images : bool
        whether to show the images or not
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: the cropped images
        
    '''

    #open the images depending on the type
    move_image= _image_to_numpy(moving_image)
    fix_image = _image_to_numpy(fixed_image)
    if isinstance(config_dict, dict):
        LOWEDGE_BOX, HIGHEDGE_BOX = config_dict.get('LOWEDGE_BOX', float(Fraction(1.5/9))), config_dict.get('HIGHEDGE_BOX', float(Fraction(7.5/9)))
    else:
        num_dict = config_dict.num_dict
        LOWEDGE_BOX, HIGHEDGE_BOX = num_dict['LOWEDGE_BOX'], num_dict['HIGHEDGE_BOX']
    cropped_moving_image = move_image[int(move_image.shape[0]*LOWEDGE_BOX):int(move_image.shape[0]*HIGHEDGE_BOX), int(LOWEDGE_BOX*move_image.shape[1]):int(HIGHEDGE_BOX*move_image.shape[1]), :]
    cropped_fixed_image = fix_image[int(LOWEDGE_BOX*fix_image.shape[0]):int(fix_image.shape[0]*HIGHEDGE_BOX), int(LOWEDGE_BOX*fix_image.shape[1]):int(HIGHEDGE_BOX*fix_image.shape[1]), :]

    

    standard_log.debug(f'the shape of the original moving image is: {move_image.shape}, the shape of the cropped moving image is: {cropped_moving_image.shape}')
    standard_log.debug(f'the shape of the original fixed image is: {fix_image.shape}, the shape of the cropped fixed image is: {cropped_fixed_image.shape}')
    return cropped_moving_image, cropped_fixed_image


    
