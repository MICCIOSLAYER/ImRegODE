# for preprocessing of various registration method

from typing import Callable, TypedDict, Optional, Tuple, Sequence, Required, NotRequired, Union, Literal
from Main_Folder.Modules.utils import permute_channel_layout
from Main_Folder.Modules.framework.data_classes import SampleDict
from Main_Folder.Modules.framework.visualization import _image_to_numpy
from Main_Folder.Modules.configuration_setting.logger_configuration import get_logger
from Main_Folder.Modules.configuration_setting.yaml_configuration import FrameworkConfig
import torch
import SimpleITK as sitk
import numpy as np
import cv2
from skimage.transform import pyramid_gaussian
from skimage.filters import gaussian
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
                
                )-> Tuple[torch.Tensor, torch.Tensor]:
    '''
    given the images it crop to the desired size 
    if requested the images are shown as confront between the original and the cropped
    
    Parameters
    ------------
    moving_image : Path | torch.Tensor
        the image to crop, it can be a path or a tensor given by the dataloader
    fixed_image : Path | torch.Tensor
        the image to crop, it can be a path or a tensor given by the dataloader 
    show_images : bool
        whether to show the images or not
        
    Returns
    --------
        Tuple[torch.Tensor, torch.Tensor]: the cropped images tuple (test_image_cropped, reference_image_cropped)
        
    '''

    #open the images depending on the type
    move_image= _image_to_numpy(moving_image)
    fix_image = _image_to_numpy(fixed_image)
    if isinstance(config_dict, dict):
        LOWEDGE_BOX, HIGHEDGE_BOX = config_dict.get('LOWEDGE_BOX', float(Fraction(1.5/9))), config_dict.get('HIGHEDGE_BOX', float(Fraction(7.5/9)))
    else:
        num_dict = config_dict.num_dict
        LOWEDGE_BOX, HIGHEDGE_BOX = num_dict['LOWEDGE_BOX'], num_dict['HIGHEDGE_BOX']
    cropped_moving_image = torch.from_numpy(move_image[int(move_image.shape[0]*LOWEDGE_BOX):int(move_image.shape[0]*HIGHEDGE_BOX), int(LOWEDGE_BOX*move_image.shape[1]):int(HIGHEDGE_BOX*move_image.shape[1]), :])
    cropped_fixed_image = torch.from_numpy(fix_image[int(LOWEDGE_BOX*fix_image.shape[0]):int(fix_image.shape[0]*HIGHEDGE_BOX), int(LOWEDGE_BOX*fix_image.shape[1]):int(HIGHEDGE_BOX*fix_image.shape[1]), :])

    

    standard_log.debug(f'the shape of the original moving image is: {move_image.shape}, the shape of the cropped moving image is: {cropped_moving_image.shape}')
    standard_log.debug(f'the shape of the original fixed image is: {fix_image.shape}, the shape of the cropped fixed image is: {cropped_fixed_image.shape}')
    return cropped_moving_image, cropped_fixed_image


    

def crop_image_advance(
        original_image: torch.Tensor | np.ndarray,
        config_dict : dict | FrameworkConfig,
        crop_mode: Optional[Literal['area_proportion', 'area_ratio','keep_props']] = None,
        
        new_O_O : Optional[Sequence[int]]= None,
        new_image_center : Optional[Sequence[int]]= None,
        centred : bool= True,
        crop_style :  Literal['keep_size', 'shrink_to_fit'] = 'keep_size',
        
        
        )-> Tuple[dict, torch.Tensor]:
    '''
    the function purpouse is to crop the image in 3 different avaiable ways: centred along original image, keeping proportions, not centred

    Args:
        original_image (Path | torch.Tensor | np.ndarray): The image must be in the shape format ( C, H, W)
        final_height (int): the desired heigh of cropped image
        final_width (int): the desired width of cropped image
        area_ratio (Optional[float], optional): the area ratio between the original and the final image, to be defined only in case of keep_proportion is true
        new_O_O (Optional[[int]], optional): the new left high corner of the image useful to the trasformation of decropped H
        new_image_center (Optional[[int]], optional): to define the cropped area starting from a center point
        centred (Optional[centred_dict], optional): to define if the cropped image centre and the original image centre are the same
        crop_style ( Literal): define if keeps the dimension of the crop as costant sliding if outside the box or stop at the box, modifing the final_height and final_width
        
        config_dict(dict): the dict from the yaml configuration

    Returns:
    Tuple of a cropped dict and a cropped image as a tensor of 
    A crop_dict={'h_f': final_height,
                   'w_f': final_width,
                   'image_center': image_center_coords,
                   'corner_coords': new_O_O new coordinate respecting the original image dimension of the high left corner,useful to the trasformation of decropped H
 
    A cropped_image (np.ndarray): the cropped image as a tensor of shape (1, C, h1, w1)
    '''
    # define dict for crop, depending on type
    if isinstance(config_dict, dict):
        crop_dict = config_dict
    else:
        crop_dict = config_dict.num_dict['crop_dict']
    
    final_height = crop_dict['final_height']
    final_width = crop_dict['final_width']
    area_ratio = crop_dict.get('area_ratio', 0)
    if isinstance(area_ratio, str) and '/' in area_ratio:
        area_ratio = float(Fraction(area_ratio))
    standard_log.debug(f'the measure chosen for the crop\nfinal_height: {final_height}; final_width: {final_width}')
    if new_O_O == None:
        new_O_O = [0, 0]

    if isinstance(original_image, np.ndarray): #NOTE shape it has to be in (C, H, W)
        original_image = permute_channel_layout(image=torch.from_numpy(original_image).float(), target_format = 'C**')

        if original_image.ndim==3:
            original_image=original_image.unsqueeze(0)

    
    
    # NOTE to be used on image after taken by dataset/dataloader in shape (1, C, H, W)

    
    # =============HANDLE  MAX DIMENSIONS============
    if (original_image.squeeze()).ndim == 3:
        _, original_height, original_width = original_image.squeeze().shape
    else:
        original_height, original_width = original_image.squeeze().shape

    original_center = [original_height//2, original_width//2]
    standard_log.debug(f'\noriginal_center: [{original_center[0]}, {original_center[1]}]')

    final_width = min(final_width, original_width)
    final_height = min(final_height, original_height)
    standard_log.debug(f'\nafter asdjusting the measure\nfinal_height: {final_height}; final_width: {final_width}')

    # ============= HANDLE DIFFERENT OPTIONS============
    
    if crop_mode=='area_proportion': # NOTE General Case
        area_ratio = area_ratio if 0 < area_ratio <=1 else 1.0
        final_height = original_height*np.sqrt(area_ratio)
        final_width = original_width*np.sqrt(area_ratio)
        standard_log.debug(f'\nafter adjusting the measure considering the {crop_mode.upper()}\nfinal_height: {final_height}; final_width: {final_width}')

    elif crop_mode=='keep_props': # Mantain proportion between original & finals
        if final_width > final_height:
            final_height = original_height*final_width/original_width
        else:
            final_width = original_width*final_height/original_height
        standard_log.debug(f'\nafter adjusting the measure considering the {crop_mode.upper()}\nfinal_height: {final_height}; final_width: {final_width}')

    elif crop_mode=='area_ratio': # Set the final dimension depend from area ratio
        area_ratio = area_ratio if 0 < area_ratio <=1 else 1.0

        if final_width > final_height:
            final_height = original_height*original_width*area_ratio/(final_width)
        elif final_width < final_height:
            final_width = original_height*original_width*area_ratio/(final_height)
        else: # NOTE use the case for keep proportions and area ratio
            final_height = original_height*np.sqrt(area_ratio)
            final_width = original_width*np.sqrt(area_ratio)
        standard_log.debug(f'\nafter adjusting the measure considering the {crop_mode.upper()} \nfinal_height: {final_height}; final_width: {final_width}')


    final_width= round(final_width)    
    final_height= round(final_height)
    standard_log.debug(f'\nafter rounding: rounded final H : {final_height}, W: {final_width}')
    # GET coordinatio of the left high point from center 
    if new_image_center != None:

        new_dim = [final_height//2, final_width//2]
        new_O_O = [ max(0,i-j) for i, j in zip(new_image_center, new_dim)]

    elif centred: 
        new_dim = [final_height//2, final_width//2]
        new_O_O = [ max(0, i-j) for i, j in zip(original_center, new_dim)]
    standard_log.debug(f'\nnew_high left corner: x0 {new_O_O[0]}, y0 {new_O_O[1]}')

    #===== GENERATE CROPPED IMAGE FROM ORIGINAL===========
    y0, x0 = new_O_O # new left high corner of image
    
    if crop_style == 'keep_size': # shift crop if necessary to keep the size of area requested for the crop
        x0 = max(0, min(x0, int(original_width-final_width)))
        y0 = max(0, min(y0, int(original_height-final_height)))
        x1 = x0 + final_width
        y1 = y0 + final_height
        standard_log.debug(f'\nconsidering box adjustment to keep size of HxW:  {final_height} x {final_width}\n x0: {x0}, y0: {y0}, x1: {x1}, y1: {y1}')
        
    elif crop_style == 'shrink_to_fit': # sacrifice a part of area since outside the original image dimensions
        x1 = x0 + final_width
        y1 = y0 + final_height
        x1 = min(x1, original_width)
        y1 = min(y1, original_height)
        standard_log.debug(f'\nconsidering box adjustment to shrink to fit for {final_width} x {final_height}\n x0: {x0}, y0: {y0}, x1: {x1}, y1: {y1}')
        
    image_center_coords = [ y0 + final_height//2, x0 + final_width//2]
    cropped_image = original_image[:, :, y0:y1, x0:x1]
    new_O_O = [y0, x0]
    crop_dict={'h_f': final_height,
               'w_f': final_width,
               'image_center': image_center_coords,
               'corner_coords': new_O_O

    }
    return crop_dict, cropped_image


# ==============
# NORMALIZE FN
# ==============
def normalize_img (image : image_type | torch.Tensor,
                   normalization_interval: tuple[int,int] = (0,1)
                   )-> torch.Tensor | np.ndarray:
    '''
    A simple  normalization function to mimic cv2.normalize fn
    The formula to normalize the image is following: 

                        img_norm(x_norm) = (x_img - img_MIN) * ( NORM_MAX - NORM_MIN)/ (img_MAX - img_MIN) + NORM_MIN


    Args:
        image (image_type | torch.Tensor): input image to normalize
        normalization_interval (tuple[int,int], optional): the final min and max of the image values. Defaults to (0,1).

    Returns:
        torch.Tensor | np.ndarray: the normalize image
    '''
    NORM_MIN = min(normalization_interval)
    NORM_MAX = max(normalization_interval)
    if isinstance(image, torch.Tensor):
        img_MIN, img_MAX = image.min(), image.max()
        norm_image = (image - img_MIN)*(NORM_MAX - NORM_MIN)/ (img_MAX -img_MIN)
        return norm_image
    else: 
        return cv2.normalize(_image_to_numpy(image), None, alpha=NORM_MIN, beta=NORM_MAX, norm_type = cv2.NORM_MINMAX, dtype=cv2.CV_32F)

#=============
# PYRAMID
#=============


def coupled_gaussian_pyramid(sample_dict: SampleDict,
                             config_dict : dict | FrameworkConfig,
                             )-> SampleDict:
    '''
    it takes two images to performa gaussian filter in 6 progressive steps
    Parameter
    ---------
    sample_dict (SampleDict): the dict with sample's  data 
    config_dict (dict): the dict with configuration parameters

    Returns
    -------
    SampleDict: updated sample dcit with the pyramids of images for both test and reference images

    '''   
    
    fixed_image = sample_dict['reference_sample']
    moving_image = sample_dict['test_sample']
    drmine_pyramid_dict = config_dict['registrations']['DRMINE_original']['pyramid']
    downscale = drmine_pyramid_dict['gaussian_downscale']
    gaussian_sigma = drmine_pyramid_dict['gaussian_sigma']
    if np.ndim(fixed_image) != np.ndim(moving_image) or type(fixed_image) != type(moving_image):
      raise ValueError("the two images must have the same type and dimension")
    else:

        if isinstance(fixed_image, torch.Tensor):
            
            fixed_image = fixed_image.squeeze().permute(1,2,0)
            moving_image = moving_image.squeeze().permute(1,2,0)
        if np.ndim(fixed_image) == 3:
            standard_log.debug(f'the shape of the fixed image is: {fixed_image.shape}, the shape of the moving image is: {moving_image.shape}')
            
            channel_axis = torch.argmin(torch.tensor(fixed_image.shape)).item()
            nChannel = fixed_image.shape[channel_axis]
            standard_log.info(f'the number of channels in the images is: {nChannel}')
            pyramid_fixed = tuple(torch.from_numpy(pyr) for pyr in pyramid_gaussian(gaussian(fixed_image, sigma=gaussian_sigma, channel_axis=channel_axis), downscale=downscale, channel_axis=channel_axis)) # NOTE multichannel = True?
            pyramid_moving = tuple(torch.from_numpy(pyr) for pyr in pyramid_gaussian(gaussian(moving_image, sigma=gaussian_sigma, channel_axis=channel_axis), downscale=downscale, channel_axis=channel_axis)) # NOTE multichannel = True?
        elif np.ndim(fixed_image) == 2:
            nChannel=1
            pyramid_fixed = tuple(torch.from_numpy(pyr) for pyr in pyramid_gaussian(gaussian(fixed_image, sigma=gaussian_sigma, channel_axis=None), downscale=downscale)) # NOTE multichannel = False?
            pyramid_moving = tuple(torch.from_numpy(pyr) for pyr in pyramid_gaussian(gaussian(moving_image, sigma=gaussian_sigma, channel_axis=None), downscale=downscale)) # NOTE multichannel = False?
        else:
            standard_log.warning(f"Unknown rank for an image: {np.ndim(fixed_image)}")

    sample_dict['reference_pyramid'] = pyramid_fixed
    sample_dict['test_pyramid'] = pyramid_moving
    sample_dict['nChannel'] = nChannel
    return sample_dict
