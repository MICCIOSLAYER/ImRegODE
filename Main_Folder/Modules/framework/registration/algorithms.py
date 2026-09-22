# all registration algorithms, it contains the complete wraps of registration, 
# final products registration that give out result to be collected
from typing import Sequence, Callable, Optional, Union, Any, Tuple
from Main_Folder.Modules.framework.registration.methods import initialize_networks, initialize_optimizer
import Main_Folder.Modules.framework.registration.networks 
from Main_Folder.Modules.utils import block_time, concatenate_paths, get_flatten_dict, has_required_keys
from Main_Folder.Modules.framework.metrics import metric_outputs_update
from Main_Folder.Modules.framework.registration.loops import pyramid_loss
from Main_Folder.Modules.framework.data_classes import SampleDict, Registration_Data_Collector
from Main_Folder.Modules.configuration_setting.yaml_configuration import FrameworkConfig
from Main_Folder.Modules.framework.postprocessing import get_affine_matrix_from_sitk_transform
from Main_Folder.Modules.framework.preprocessing import general_preprocessing, airlab_mask_configuration
from Main_Folder.Modules.framework.visualization import airlab_show_image_differencies
from Main_Folder.Modules.configuration_setting.logger_configuration import  get_logger
from Main_Folder.Modules.framework.img_io  import tensor_img_to_sitk, airlab_read_image
from Main_Folder.Modules.framework.evaluations import naed_evaluation, update_with_evaluation, wrapper_naed_drmine
import SimpleITK as sitk
from airlab.utils.image import Image as AirlabImage
from airlab.registration import PairwiseRegistration
from airlab.transformation.pairwise import AffineTransformation
from airlab.loss.pairwise import MI
from airlab.transformation.utils import warp_image
from pathlib import Path
from tqdm.auto import tqdm
from itertools import islice
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch
import itk
import numpy as np
import sys


standard_log = get_logger(__name__)
_fwc_dict = FrameworkConfig()
_default_save_dir = _fwc_dict.path_dict['DRIVE_RESULTS'] if 'google.colab' in sys.modules else _fwc_dict.path_dict['RESULTS'] 




#                                    =================================
#                                            SITK REGISTRATION
#                                    =================================

#-------------------------------------------WRAPPER-------------------------------------------

def sitk_wrapper_for_registration(sample_dict: SampleDict, 
                              config_dict: dict,
                              sitk_registration_fn: Callable[[sitk.Image, sitk.Image, dict], tuple[Any, sitk.ImageRegistrationMethod, float]],
                              
                              )-> dict:
    '''
    A wrapper function to run the registration pipeline using SimpleITK. It takes a sample dictionary and a configuration dictionary as input and returns a dictionary containing the registration results.

    Parameters
    -------------
        sample_dict (SampleDict): A dictionary containing the reference and test samples, their paths, and the sample name.
        config_dict (dict): A dictionary containing configuration parameters for registration.
        sitk_registration_fn (Callable): A function that performs the registration using SimpleITK:
            INPUTs - sitk.Image, sitk.Image, dict
            OUTPUTs - tuple[outTx{Any}, sitk.ImageRegistrationMethod, float]

    Return
    ------------
        dict (dict): A dictionary containing the registration results, including the affine matrix, NAED value, and time taken for registration, mean for used in the data collector
    '''
    
    reference_image = sample_dict['reference_sample']
    test_image = sample_dict['test_sample']

    if isinstance(reference_image, Path):
        reference_sitk = sitk.ReadImage(reference_image, sitk.sitkFloat32)
    elif isinstance(reference_image, torch.Tensor) or isinstance(reference_image, np.ndarray):
        reference_sitk = tensor_img_to_sitk(reference_image)

    if isinstance(test_image, Path):
        test_sitk = sitk.ReadImage(test_image)
    elif isinstance(test_image, torch.Tensor) or isinstance(test_image, np.ndarray):
        test_sitk = tensor_img_to_sitk(test_image)

    if reference_sitk.GetSize() != test_sitk.GetSize():
            raise ValueError("The two images must have the same shape")
    
    #NOTE registration_data is not accessed
    
    sitk_transformation, registration_data, time_taken = sitk_registration_fn(reference_image = reference_sitk,
                                                                            test_image = test_sitk,
                                                                            config_dict = config_dict)
    
    #NOTE remember to use registration_data to get an affine matrix from get_affine_matric_from_sitk_transform
    homography_matrix=get_affine_matrix_from_sitk_transform(sitk_transform=sitk_transformation)

    naed_diagonal = naed_evaluation(image_couple_name=sample_dict['sample_name'],
                                    affine_matrix=homography_matrix,
                                    image_normalization='diagonal')
    naed_norm_coord = naed_evaluation(image_couple_name=sample_dict['sample_name'],
                                    affine_matrix=homography_matrix,
                                    image_normalization='coords_norm')
    

    registration_results={}
 
    #registration_results['registration_object']= registration_data NOTE not useful now
    registration_results['naed_diagonal'] = naed_diagonal
    registration_results['naed_coords'] = naed_norm_coord
    registration_results['sitk_transformation'] =  sitk_transformation
    registration_results['affine_matrix'] = homography_matrix
    registration_results['time_taken'] = time_taken
    
    return registration_results


#-------------------------------------------REGISTRATION FNS-------------------------------------------


def MMI_SITK(reference_image: sitk.Image,
            test_image: sitk.Image,
            config_dict : dict | FrameworkConfig = None,
              )->tuple[Any, sitk.ImageRegistrationMethod, float]: # NOTE see the output
    '''
    Mattes Mutual Information (MMI) between two images using SimpleITK.
    
    Parameters
    ----------
    reference_image : np.ndarray | torch.Tensor
        The reference image.
    test_image : np.ndarray | torch.Tensor
        The test image.
    config_dict : dict | FrameworkConfig
        If the dict, it has to have the following keys at least: 
        ['histo_bins', 'sampling_ratio', 'lr', 'n_iterations', 'convergenceMinimumValue', 'convergenceWindowSize']
    
    Required Keys 
    ---------------
    'sampling_ratio', 'histo_bins', 'lr', 'n_iterations', 'convergenceMinimumValue', 'convergenceWindowSize'
    
    Returns
    -------
    outTx : sitk.Transform
        The resulting transformation to easly access the trasformation through sitk methods
    registration : sitk.ImageRegistrationMethod
        The registration object containing all the registration parameters and methods
    registration_time_taken : float
        The time taken for the registration process.
    '''
    required_keys =['sampling_ratio', 'histo_bins', 'lr', 'n_iterations', 'convergenceMinimumValue', 'convergenceWindowSize']
    if isinstance(config_dict, dict):
        if has_required_keys(input_dict=config_dict, required_keys=required_keys):
            mmi_sitk_dict = config_dict
        else:
            standard_log.warning('the config_dict miss some of required_keys, so the default is used, instead')
            mmi_sitk_dict= _fwc_dict.registrations['simpleITK_original']['MMI']
    else:
        mmi_sitk_dict = config_dict.registrations['simpleITK_original']['MMI']
    
    
    samplingPercentage = float(mmi_sitk_dict['sampling_ratio'])
    histogram_bins = int(mmi_sitk_dict['histo_bins'])
    lr = float(mmi_sitk_dict['lr'])
    n_iterations = float(mmi_sitk_dict['n_iterations'])
    convergence_min_value = float(mmi_sitk_dict['convergenceMinimumValue'])
    convergence_window_size = float(mmi_sitk_dict['convergenceWindowSize'])
    # to get the transformation of the matrix
    with block_time() as mattes_time:
        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=int(histogram_bins))  
        registration.SetMetricSamplingPercentage(float(samplingPercentage), seed=sitk.sitkWallClock) 
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetOptimizerAsGradientDescent(learningRate=float(lr), numberOfIterations=int(n_iterations), convergenceMinimumValue=float(convergence_min_value), convergenceWindowSize=int(convergence_window_size) )
        registration.SetOptimizerScalesFromPhysicalShift() # NOTE added as online suggestions
        registration.SetInitialTransform(sitk.AffineTransform(reference_image.GetDimension()))  # Initial transform
        registration.SetInterpolator(sitk.sitkLinear)  # Interpolator

        #registration.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration)) # to get the iteration information while executing the cycle

        # to get the transformation of the warped image:
        outTx = registration.Execute(reference_image, test_image)  # Execute the registration
    registration_time_taken = mattes_time[0]
    # get information about the iteration cycle:

    return outTx, registration, registration_time_taken



def centred_NCC_SITK(reference_image: np.ndarray | torch.Tensor | Path,
                     test_image: np.ndarray | torch.Tensor | Path,
                     config_dict:dict|FrameworkConfig,)-> tuple: # NOTE see the output
    '''
    Mattes Mutual Information (MMI) between two images using SimpleITK.
    
    Parameters
    ----------
    reference_image : np.ndarray | torch.Tensor
        The fixed image.
    test_image : np.ndarray | torch.Tensor
        The warped image.

    Required Keys 
    ---------------
    'lr','min_step', 'n_iterations', 'gradientMagnitudeTolerance'
    
    Returns
    -------
    float
        The Normalized Mutual Information value.
    '''
    required_keys= ['lr','min_step', 'n_iterations', 'gradientMagnitudeTolerance']
    if isinstance(config_dict, dict):
            if has_required_keys(input_dict=config_dict, required_keys=required_keys):
                ncc_centred_dict = config_dict
            else:
                ncc_centred_dict=_fwc_dict.registrations['simpleITK_original']['NCC_centred']
    else:
        ncc_centred_dict = config_dict.registrations['simpleITK_original']['NCC_centred']
    
    
    with block_time() as ncc_time:
        # to get the transformation of the matrix
        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsCorrelation()  
        registration.SetOptimizerAsRegularStepGradientDescent(
            learningRate=ncc_centred_dict['lr'],
            minStep=ncc_centred_dict['min_step'],
            numberOfIterations=ncc_centred_dict['n_iterations'],
            gradientMagnitudeTolerance=ncc_centred_dict['gradientMagnitudeTolerance'],
        )
        registration.SetOptimizerScalesFromIndexShift()
        
        transformation = sitk.CenteredTransformInitializer( reference_image, test_image, sitk.AffineTransform(reference_image.GetDimension())) # TRY to use operation_mode = MOMENTS
        registration.SetInitialTransform(transformation)

        registration.SetInterpolator(sitk.sitkLinear)  # Interpolator
        out_tx = registration.Execute(reference_image, test_image)  # Execute the registration
    time_taken = ncc_time[0]
    

    # get information about the iteration cycle:

    return out_tx, registration, time_taken



def JHMI_SITK(reference_image: sitk.Image,
              test_image: sitk.Image,
              config_dict:dict| FrameworkConfig,
              )->tuple[Any, sitk.ImageRegistrationMethod, float]: # NOTE see the output
    '''
    Mattes Mutual Information (MMI) between two images using SimpleITK.
    
    Parameters
    ----------
    reference_image : np.ndarray | torch.Tensor
        The reference image.
    test_image : np.ndarray | torch.Tensor
        The test image.

    Required Keys
    ------------
    'sampling_ratio', 'lr', 'n_iterations', 'convergenceMinimumValue', 'convergenceWindowSize','histo_bins'
        
    Returns
    -------
    float
        The Normalized Mutual Information value.
    '''
    req_ks = ['sampling_ratio', 'lr', 'n_iterations', 'convergenceMinimumValue', 'convergenceWindowSize','histo_bins']
    if isinstance(config_dict, dict) :
        if has_required_keys(input_dict=config_dict, required_keys=req_ks):
            jhmi_sitk_dict = config_dict
        else:
            standard_log.warning('the config_dict miss some of required_keys, so the default is used, instead')
            jhmi_sitk_dict= _fwc_dict.registrations['simpleITK_original']['JHMI']
        
    else:
        jhmi_sitk_dict=config_dict.registrations['simpleITK_original']['JHMI']
    # normalization & Gaussian smoothing:
    #reference_image = sitk.DiscreteGaussian(sitk.Normalize(reference_image), 2.0) #removed since not present in the article
    #test_image = sitk.DiscreteGaussian(sitk.Normalize(test_image), 2.0) #removed since not present in the article

    
    samplingPercentage = jhmi_sitk_dict['sampling_ratio']
    lr = jhmi_sitk_dict['lr']
    n_iterations = jhmi_sitk_dict['n_iterations']
    convergence_min_value = jhmi_sitk_dict['convergenceMinimumValue']
    convergence_window_size = jhmi_sitk_dict['convergenceWindowSize']
    histo_bins = jhmi_sitk_dict['histo_bins']
    
    with block_time() as jhmi_time:
    # to get the transformation of the matrix
        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=histo_bins)  
        registration.SetOptimizerAsGradientDescent( #NOTE try also with SetOptimzerAsGradientDescent otherwise it will take 16min
            learningRate=lr,
            numberOfIterations=n_iterations,
            convergenceMinimumValue=convergence_min_value,
            convergenceWindowSize=convergence_window_size,
        )
        registration.SetInitialTransform(sitk.AffineTransform(reference_image.GetDimension()))  # Initial transform
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(samplingPercentage)  # 50% sampling
        registration.SetOptimizerScalesFromPhysicalShift() # NOTE added as online suggestions
        registration.SetInterpolator(sitk.sitkLinear)  # Interpolator
        out_tx = registration.Execute(reference_image, test_image)  # Execute the registration 
    time_taken = jhmi_time[0]  

    return out_tx, registration, time_taken



# Mean Squared Error:
def MSE_SITK( reference_image: sitk.Image,
              test_image: sitk.Image,
              config_dict : dict |FrameworkConfig,
              )->tuple[Any, sitk.ImageRegistrationMethod, float]: # NOTE see the output
    '''
    Mean Squared Error (MMI) between two images using SimpleITK.
    
    Parameters
    ----------
    fixed_image : np.ndarray | torch.Tensor
        The fixed image.
    warped_image : np.ndarray | torch.Tensor
        The warped image.

    Required Keys
    ------------
    'sampling_ratio', 'lr', 'n_iterations', 'convergenceMinimumValue', 'convergenceWindowSize'
    
    Returns
    -------
    float
        The Normalized Mutual Information value.
    '''
    req_ks = ['sampling_ratio', 'lr', 'n_iterations', 'convergenceMinimumValue', 'convergenceWindowSize']
    if isinstance(config_dict, dict) :
        if has_required_keys(input_dict=config_dict, required_keys=req_ks):
            mse_sitk_dict = config_dict
        else:
            standard_log.warning('the config_dict miss some of required_keys, so the default is used, instead')
            mse_sitk_dict= _fwc_dict.registrations['simpleITK_original']['MSE']
    else:
        mse_sitk_dict=config_dict.registrations['simpleITK_original']['MSE']
    
    samplingPercentage = mse_sitk_dict['sampling_ratio']
    lr = mse_sitk_dict['lr']
    n_iterations = mse_sitk_dict['n_iterations']
    convergence_min_value = mse_sitk_dict['convergenceMinimumValue']
    convergence_window_size = mse_sitk_dict['convergenceWindowSize']
    
    with block_time() as mse_time:
        # to get the transformation of the matrix
        registration = sitk.ImageRegistrationMethod()

        registration.SetMetricAsMeanSquares()  
        registration.SetOptimizerAsGradientDescent(
            learningRate = lr,
            numberOfIterations = n_iterations,
            convergenceMinimumValue = convergence_min_value,
            convergenceWindowSize = convergence_window_size,

        )
        registration.SetInitialTransform(sitk.AffineTransform(reference_image.GetDimension()))  # Initial transform
        registration.SetOptimizerScalesFromPhysicalShift()
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(samplingPercentage)  # 50% sampling
        registration.SetInterpolator(sitk.sitkLinear)  # Interpolator
        out_tx = registration.Execute(reference_image, test_image)  # Execute the registration   
    time_taken = mse_time[0]
    return out_tx, registration, time_taken



def NCC_SITK(reference_image: sitk.Image,
             test_image:  sitk.Image,
             config_dict: dict|FrameworkConfig,
           )-> tuple[Any, sitk.ImageRegistrationMethod, float]: # NOTE see the output
    '''
    Registration of two images using Normalized Cross Correlation (NCC) of SimpleITK as metric.
    
    Parameters
    ----------
    fixed_image : np.ndarray | torch.Tensor
        The fixed image.
    warped_image : np.ndarray | torch.Tensor
        The warped image.

    Required Keys
    ------------
    'sampling_ratio', 'lr', 'n_iterations', 'convergenceMinimumValue', 'convergenceWindowSize'

    Returns
    -------
    float
        The normalized cross correlation value.
    '''
    req_ks = ['sampling_ratio', 'lr', 'n_iterations', 'convergenceMinimumValue', 'convergenceWindowSize']
    if isinstance(config_dict, dict) :
        if has_required_keys(input_dict=config_dict, required_keys=req_ks):
            ncc_sitk_dict = config_dict
        else:
            standard_log.warning('the config_dict miss some of required_keys, so the default is used, instead')
            ncc_sitk_dict= _fwc_dict.registrations['simpleITK_original']['NCC']
    else:
        ncc_sitk_dict=config_dict.registrations['simpleITK_original']['NCC']
    
    
    samplingPercentage = ncc_sitk_dict['sampling_ratio']
    lr = ncc_sitk_dict['lr']
    n_iterations = ncc_sitk_dict['n_iterations']
    convergence_min_value = ncc_sitk_dict['convergenceMinimumValue']
    convergence_window_size = ncc_sitk_dict['convergenceWindowSize']
    # to get the transformation of the matrix
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsCorrelation()  # NCC metric
    registration.SetMetricSamplingPercentage(samplingPercentage, seed=sitk.sitkWallClock) # NOTE see if optional or a must
    registration.SetMetricSamplingStrategy(registration.RANDOM) #NOTE not defined in the article
    registration.SetOptimizerAsGradientDescent(learningRate=lr, convergenceMinimumValue=convergence_min_value, convergenceWindowSize=convergence_window_size, numberOfIterations=n_iterations) 
    registration.SetOptimizerScalesFromPhysicalShift() # NOTE added as online suggestions
    registration.SetInitialTransform(sitk.AffineTransform(reference_image.GetDimension()))  # Initial transform
    registration.SetInterpolator(sitk.sitkLinear)  # Interpolator

    #    registration.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration)) # to get the iteration information while executing the cycle
    with block_time() as ncc_time:
        # to get the transformation of the warped image:
        outTx = registration.Execute(reference_image, test_image)  # Execute the registration
    time_taken = ncc_time[0]
    return outTx, registration, time_taken



def ncc_sitk_chatgpy(reference_image: np.ndarray | torch.Tensor | Path | sitk.Image,
           test_image: np.ndarray | torch.Tensor | Path | sitk.Image,
           transform : sitk.Transform

           )-> tuple:
    '''
    Normalized Cross Correlation (NCC) between two images using SimpleITK.
    
    Parameters
    ----------
    fixed_image : np.ndarray | torch.Tensor
        The fixed image.
    warped_image : np.ndarray | torch.Tensor
        The warped image.
    
    Returns
    -------
    float
        The normalized cross correlation value.
    '''

        
    ncc_filter = sitk.FFTNormalizedCorrelationImageFilter()
    with block_time() as ncc_time:
        ncc_map = ncc_filter.Execute(reference_image, test_image)

        stats = sitk.StatisticsImageFilter()
        stats.Execute(ncc_map)
    time_taken = ncc_time[0]
    return stats.GetMaximum(), ncc_filter,  time_taken  # Return the maximum NCC value






#                                    =================================
#                                           ELASTIX REGISTRATION
#                                    =================================

#-------------------------------------------WRAPPER-------------------------------------------

def elastix_wrapper_for_registration(sample_dict: SampleDict,
                                     config_dict: dict | FrameworkConfig,
                                     elastix_registration_fn: Callable[[SampleDict, dict], dict],
                                     )-> dict:
    '''
    Wrapper function for elastix registration. It takes a sample dictionary and a configuration dictionary, and
    returns a dictionary containing the registration results and parameters.
    Parameters
    ----------
    sample_dict : SampleDict
        The sample dictionary containing the image paths.
    config_dict : dict, FrameworkConfig
        The configuration dictionary for the registration.
    
    Returns
    -------
    dict
        A dictionary containing the registration results and parameters.
    '''
    
    test_image = itk.imread(sample_dict['test_sample_path'], itk.F)
    reference_image = itk.imread(sample_dict['reference_sample_path'], itk.F)
    


    itk_registration, time_taken, registred_image = elastix_registration_fn(reference_image=reference_image,
                                                           test_image=test_image,
                                                           config_dict=config_dict)
    homography_matrix = get_affine_matrix_from_sitk_transform(sitk_transform=itk_registration)

    naed_diagonal = naed_evaluation(image_couple_name=sample_dict['sample_name'],
                                    affine_matrix=homography_matrix,
                                    image_normalization='diagonal')
    naed_norm_coord = naed_evaluation(image_couple_name=sample_dict['sample_name'],
                                    affine_matrix=homography_matrix,
                                    image_normalization='coords_norm')
    
    registration_results={}
    registration_results['registered_image']= registred_image
    registration_results['homography_matrix'] = homography_matrix
    registration_results['time_taken'] = time_taken
    registration_results['naed_diagonal'] = naed_diagonal
    registration_results['naed_norm_coord'] = naed_norm_coord

    return registration_results




#-------------------------------------------REGISTRATION FNS-------------------------------------------

def elastix_registration(reference_image: itk.Image,
                        test_image: itk.Image,
                        config_dict: dict | FrameworkConfig,
                       
                        )-> tuple:
    '''
    elastix registration for NMI metric algorithm

    Args
    -----
        reference_image (itk.Image): itk image of reference
        test_image (itk.Image): itk image of test as deformed
        config_dict (dict | FrameworkConfig): configuration object:
            DICT -> mandatory keys: 'sampling_ratio', 'histo_bins', 'n_resolution', 'n_iterations'
    
    Required Keys
    ------------
    'sampling_ratio', 'histo_bins', 'n_resolution', 'n_iterations'
    
    Returns
    -------
    tuple
        The registered image and the registration parameters.
    '''
    req_ks =['sampling_ratio', 'histo_bins', 'n_resolution','n_iterations']
    if isinstance(config_dict, dict) :
        if has_required_keys(input_dict=config_dict, required_keys=req_ks):
            elastix_config_dict = config_dict
        else:
            standard_log.warning('the config_dict miss some of required_keys, so the default is used, instead')
            elastix_config_dict= _fwc_dict.registrations['simpleITK_original']['JHMI']
    else:
        elastix_config_dict = config_dict.registrations['elastix_original']
    
    sampling_ratio = elastix_config_dict['sampling_ratio'] # not specified in the article, but used to be coherent with other methods
    histogram_bins= elastix_config_dict['histo_bins']
    num_resolutions= elastix_config_dict['n_resolution']
    max_iterations= elastix_config_dict['n_iterations']
    with block_time() as registration_time:
        transformation_parameters = itk.ParameterObject.New() # set the pparameter object
        affine_params_map = transformation_parameters.GetDefaultParameterMap('affine') # set the parameter for affine transformation
        affine_params_map['metric'] = ['NormalizedMutualInformation']
        affine_params_map['MetricSamplingPercentage'] = [f'{sampling_ratio}']
        affine_params_map['WritingResultImage'] = ['false'] # to avoid saving the registered image in the current folder
        affine_params_map['DefauldtPixelValue'] = ['0'] # to avoid having black borders in the registered image
        affine_params_map['NumberOfHistogramBins'] = [f'{histogram_bins}']
        affine_params_map['NumberOfResolutions'] = [f'{num_resolutions}']
        affine_params_map['MaximumNumberOfIterations'] = [f'{max_iterations}']
        transformation_parameters.SetParameterMap(affine_params_map)
    
        registred_image, registration_parameters=  itk.elastix_registration_method(
            fixed_image=reference_image,
            moving_image=test_image,
            parameter_object=transformation_parameters,
            log_to_console=True)
    time_taken = registration_time[0]


    return registration_parameters, time_taken, registred_image




#                                    =================================
#                                            AIRLAB REGISTRATION
#                                    =================================

#-------------------------------------------WRAPPER-------------------------------------------

def airlab_wrapprer_for_registration(sample_dict:SampleDict,
                                     config_dict:dict | FrameworkConfig,
                                     airlab_registration_fn: Callable[[AirlabImage, AirlabImage, dict], dict],
                                     show_difference : bool = False,
                                     saving_path : Path = None
                                     )-> dict:
    '''
    Wrapper forfor airlab registration loop -type it uniform registration loops to take the same inputs and give out the results 
    as a dict to be easily put in Data Collector

    Args:
        sample_dict (SampleDict): Dictionary containing main information about samples
        config_dict (dict, FrameworkConfig): configuration dictionary containing all the parameters for the registration
            if a dict mandatory to have the following keys: 'masked' with a bool value
                                                            also see the registrationfn in the standard it has to have
        airlab_registration_fn (_type_): registration function following specs of airlab
    Required Keys
    ------------
    'masked', if present it should contains parameter for the mask built
    Returns:
        dict: registration results of airlab_registration_fn
    '''
    req_ks = ['masked']
    if isinstance(config_dict, dict):
        if has_required_keys(input_dict=config_dict, required_keys=req_ks):
            airlab_config_dict = config_dict
        else:
            airlab_config_dict = _fwc_dict.registrations['airlab']
    else:
        airlab_config_dict= config_dict.registrations['airlab']

    
    #===========READ IMAGES============
    if 'reference_sample_path' in sample_dict:
        airlab_reference_image = airlab_read_image(sample_dict['reference_sample_path'])
    elif 'reference_sample' in sample_dict:
        airlab_reference_image = airlab_read_image(sample_dict['reference_sample'])

    if 'test_sample_path' in sample_dict:
        airlab_test_image = airlab_read_image(sample_dict['test_sample_path'])
    elif 'test_sample' in sample_dict:
        airlab_test_image = airlab_read_image(sample_dict['test_sample'])
    
    if airlab_config_dict['masked']:

        airlab_reference_mask, airlab_test_mask = airlab_mask_configuration(reference_image_airlab=airlab_reference_image,
                                                                test_image_airlab=airlab_test_image,
                                                                config_dict=config_dict)
        
        registration_data = airlab_registration_fn(reference_image=airlab_reference_image,
                                                   reference_mask = airlab_reference_mask,
                                                   test_mask=airlab_test_mask,
                                                   moved_image=airlab_test_image,
                                                   config_dict=config_dict,
                                                   )
    else:
        registration_data = airlab_registration_fn(reference_image=airlab_reference_image,
                                                   moved_image=airlab_test_image,
                                                   config_dict=config_dict,
                                                   )
    if show_difference:
        airlab_show_image_differencies(
            test_image=airlab_test_image,
            reference_image=airlab_reference_image,
            airlab_transformation=registration_data['airlab_transformation'],
            save_images = saving_path,
            airlab_dict=config_dict, #NOTE control why airlab_config_dcit instead of the config_dict
        )


    image_name_couple = sample_dict['sample_name']
    naed_diagonal = naed_evaluation(image_couple_name=image_name_couple,
                                affine_matrix=registration_data['H_matrix'],
                                image_normalization='diagonal')
    naed_norm_coord = naed_evaluation(image_couple_name=image_name_couple,
                                affine_matrix=registration_data['H_matrix'],
                                image_normalization='coords_norm')
    
    registration_data['naed_diagonal'] = naed_diagonal
    registration_data['naed_coords'] = naed_norm_coord
    registration_data['affine_matrix'] = registration_data['H_matrix'].detach().cpu().numpy()



    
    return registration_data



#-------------------------------------------REGISTRATION FNS-------------------------------------------
def airlab_mi_registration  (reference_image: AirlabImage,
                            test_image: AirlabImage,
                            config_dict:dict | FrameworkConfig,
                            reference_mask: Optional[AirlabImage] = None,
                            test_mask: Optional[AirlabImage] = None,
                            device: Optional[torch.device] = None,
                            EarlyStopping : bool = False
                            ) -> dict: # (displacement_field, registration_state_dict)
    '''
    Perform image registration using AirLab with Mutual Information as the loss function.

    Parameters
    ----------

    reference_image: AirlabIamge
        The reference image. In a ready format from airlab.Image
    moved_image : AilrabImage
        The tested image. In a ready format from airlab.Image
    config_dict: dict
        The dict from which the parameters has to be taken
    reference_mask: AirlabImage, optional
        The mask of the reference image, to define which point to consider during the loss calcula
    test_mask: AirlabImage, optional
        The mask of the tested image, to define which point to consider during the loss calcula
    show_plots : bool
        Whether to display plots during registration.
    save_image : bool
        Whether to save plots during registration.
    
    device : torch.device, optional
        The device to run the registration on. If None, it will use 'cuda' if available, otherwise 'cpu'.
    EarlyStopping: bool
        A flag to define how to stop the registration loop

    Required Keys
    ------------
    'histo_bins', 'metric_sigma','sampling_ratio','lr','n_iterations', 
    'early-stopping' if in use it has to have: {'patience':...,
                                                'min_delta':...}
    
    Returns
    -------

        tuple: Homography matrix and a dictionary containing the registration state and parameters.
            H, dict.keys() =['registration_type', 'time_taken', 'registrations_data', 'normalization_type',
                               'metric_num_bins', 'metric_sigma', 'learning_rate', 'num_iterations',
                               'loss_history']
    '''
    #============DEVICE SELECTION============
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    req_keys=['histo_bins', 'metric_sigma','sampling_ratio','lr','n_iterations', 'early-stopping']
    if isinstance(config_dict, dict):
            if has_required_keys(input_dict=config_dict, required_keys=req_keys):
                airlab_config_dict = config_dict
            else:
                airlab_config_dict = _fwc_dict.registrations['airlab']
    else:
        airlab_config_dict= config_dict.registrations['airlab']
    
    

    airlab_config_dict = config_dict['registrations']['airlab']
    metric_num_bins = airlab_config_dict['histo_bins']
    metric_sigma = airlab_config_dict['metric_sigma']
    spatial_sampling = airlab_config_dict['sampling_ratio']
    learning_rate = airlab_config_dict['lr']
    num_iterations = airlab_config_dict['n_iterations']

    images_dim = int(reference_image.ndim)


    #=============REGISTRATION SETUP==============
    if not airlab_config_dict['masked']:
        airlab_metric = MI(
            fixed_image=reference_image, 
            moving_image=test_image, 
            bins=metric_num_bins, 
            sigma=metric_sigma,
            spatial_samples=spatial_sampling,
        )
    else:
            
            airlab_metric = MI(
            fixed_image=reference_image, 
            fixed_mask=reference_mask,
            moving_image=test_image, 
            moving_mask=test_mask,
            #background= fixed_image_airlab.numpy()[0, 0], # prova per prendere i valori di background direttamente dall'immagine, per MI da airlab, se background non è spiecificato viene assunto dal min dell'immagine
            bins=metric_num_bins, 
            sigma=metric_sigma,
            spatial_samples=spatial_sampling,
        )

    airlab_transformation = AffineTransformation(moving_image=test_image, opt_cm=False)
    airlab_transformation.init_translation(fixed_image=reference_image)

    airlab_optimizer = torch.optim.Adam(
        airlab_transformation.parameters(), 
        lr=learning_rate, 
        amsgrad=True
        )
    
    with block_time() as registration_time:
        airlab_registration = PairwiseRegistration()
        airlab_registration.set_optimizer(airlab_optimizer)
        airlab_registration.set_number_of_iterations(num_iterations)
        airlab_registration.set_transformation(airlab_transformation)
        airlab_registration.set_image_loss([airlab_metric])
        if not EarlyStopping:
            airlab_registration.start(EarlyStopping=False, )
        else:
            patience = airlab_config_dict['early-stopping']['patience']               # quante iterazioni senza miglioramento prima di fermarsi
            min_delta = airlab_config_dict['early-stopping']['min_delta']            # minimo miglioramento considerato valido (per MI spesso 1e-7)
            best_loss = float('inf')
            patience_counter = 0
            best_state = None            # salveremo qui lo stato migliore

            loss_history = []            # per tracciare tutto

            standard_log.warning("Staring registration with EarlyStopping-mode")

            for iteration in range(num_iterations):
                
                # Esegui un passo di ottimizzazione (calcola loss + backward + update)
                loss_value = airlab_optimizer.step(airlab_registration._closure)
                
                # Converti in float scalare per confronti sicuri
                current_loss = float(loss_value.item()) if torch.is_tensor(loss_value) else loss_value
                
                loss_history.append(current_loss)
                
                # Controllo miglioramento
                if current_loss < best_loss - min_delta:
                    best_loss = current_loss
                    patience_counter = 0
                    # Salva lo stato attuale dei parametri (è sicuro, non fa deepcopy profondo)
                    best_state = airlab_transformation.state_dict().copy()  # .copy() shallow è ok per dict di tensor leaf
                    standard_log.warning(f"Iter {iteration+1:4d} | Loss: {current_loss:.6f}  (best)")
                else:
                    patience_counter += 1
                    standard_log.warning(f"Iter {iteration+1:4d} | Loss: {current_loss:.6f}  (worse)")
                    
                    if patience_counter >= patience:
                        standard_log.warning(f"Early stopping activated after {iteration+1} iteration (patience={patience})")
                        # Ripristina lo stato migliore
                        if best_state is not None:
                            airlab_transformation.load_state_dict(best_state)
                        break

                                  

    airlab_time_taken = registration_time[0]
 
    
    
        
        
    #8. return the displacement field and registration state dictionary
    transformation_matrix = airlab_transformation._compute_transformation_matrix() # da sostituire con airlab_transformation.get_transformation_matrix() 
    H=torch.eye(images_dim+1, dtype=torch.float32, device=device)
    H[:images_dim, :]=transformation_matrix
    airlab_state_dict = airlab_registration._transformation.state_dict()
    airlab_state_dict.pop('_grid', None)
                               
    registration_state_dict = {'time_taken' : airlab_time_taken,
                               'airlab_transformation': airlab_transformation,
                               'registrations_data' : airlab_state_dict,
                               'H_matrix': H.detach().cpu().numpy(),
                               'loss_history': airlab_registration.lossHistory,
                               'device': device,}
                               
    return registration_state_dict



#                                    =================================
#                                            DRMINE REGISTRATION
#                                    =================================

#-------------------------------------------WRAPPER-------------------------------------------

def wrapper_drmine_registration_loop(
        
        
        sample_dict: SampleDict,
        config_dict: dict| FrameworkConfig,
        parameter_for_registration: dict = None , # output of extract_registration_param
        model_nets : Optional[dict]= None,
        
        loss_fn : Callable[..., Union[float, Sequence[float]]]=pyramid_loss,
        network_classes : Optional[dict[str, type[nn.Module]]] = None,
        network_initialization_params : dict = None,
        early_stopping: bool = False,
        progression_bar : bool = True,      
        device : torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)-> dict:
    '''
    wrapper to organize results after setting the type of registration loop, optimizer, network used, in a dict to be used as a base for the data collector structure

    Args:
        registration_name: Optional[str] the name of the dictionary in the config_dict
        network_initializer (Callable[..., dict[str, nn.Module]]): initializer of the networks useful for the registration loop
        optimizer_initializer (Callable[]): optimizer initialization, with models and parameter of interest
        parameter_for_registration (dict): parameter extracted by extract_registration_params, it has to have: 'reference_lst', 'test_lst', 'xy_lst', 'ind_lst'. Default to {}
        sample_dict (SampleDict): input point of data, complete and ordered in a typedDict datatype
        config_dict (dict): dictionary of configuration, if different from the yaml configuaration obj: it has to have the following keys:
            'lr', 'n_iterations', 'sampling_ratio', 'patience', 'min_delta'; 
            'lr' has to be a dict with the following structure:{'MINE': 0.01, 'HomographyNet': {'vL': 0.001, 'v1': 1e-05}}
        drmine_registration_fn (Callable, optional): registration loop fn. Defaults to drmine_registration_loop.
        progression_bar (bool, optional): option to visualize the progression bar. Defaults to True.
        device (torch.device, optional): _description_. Defaults to torch.device('cuda' if torch.cuda.is_available() else 'cpu').

    Returns:
        dict: dict of results from registration and some other measurments
    '''
    req_ks = ['lr','n_iterations','sampling_ratio','patience','min_delta']
    if isinstance( config_dict, dict):
        if has_required_keys(input_dict=config_dict, required_keys=req_ks):
            drmine_registration_dict = config_dict
        else: 
            drmine_registration_dict=_fwc_dict.registrations['DRMINE_original']
    else:
        drmine_registration_dict = config_dict.registrations['DRMINE_original']
    
    
    lr_dict = drmine_registration_dict['lr']
    iterations= drmine_registration_dict['n_iterations']
    sampling= drmine_registration_dict['sampling_ratio']
    patience= drmine_registration_dict['patience']
    min_delta = drmine_registration_dict['min_delta']

    if network_initialization_params is None:
        network_initialization_params = {}

    network_initialization_params.setdefault('MINE', {})['nChannel'] = sample_dict['nChannel']

    
        
    # ========== ORGANIZE THE STRUCTURE============
    if network_classes:
        net_lst = initialize_networks(network_classes = network_classes,
                                      network_kwargs_dict=network_initialization_params)
    else:
        net_lst =  initialize_networks(network_kwargs_dict=network_initialization_params)


    loss_history = []
    patience = round(iterations/10) if patience == 0 else patience
    registration_dict = {}
    sampling = sampling if 0 < sampling <=1.0 else 0.1
    counter_patience = 0
    best_loss= -float(np.inf)
    if progression_bar:
        pbar= tqdm(range(iterations), desc='DRMINE registration')
    else:
        pbar = range(iterations)


    #=========== GET THE USEFUL PARAMS FOR THE TRAINING LOOP================== 
    if parameter_for_registration is None:
        from Main_Folder.Modules.framework.registration.loops import extract_registration_params

        parameter_for_registration = extract_registration_params(sample_dict=sample_dict,
                                                                 config_dict=config_dict,
                                                                 device=device)
        
    sample_dict['reference_pyramid'] = parameter_for_registration['reference_lst']
    sample_dict['test_pyramid'] = parameter_for_registration['test_lst']
    xy_lst = parameter_for_registration['xy_lst']
    ind_lst = parameter_for_registration['ind_lst']

    parameter_for_registration['trasformation_network'] = net_lst['trasformation']
    parameter_for_registration['metric_network']= net_lst['metric']
    
    # NOTE in case of other type of loss_fn add here in parameter_for_registration


    optim_init =  initialize_optimizer(trasformation_model=net_lst['trasformation'],
                                      metric_model = net_lst['metric'],
                                      trasformation_lr=lr_dict[type(net_lst['trasformation']).__name__], # NOTE net_lst contain objects already instantiated, so type(..) is compulosry
                                      metric_lr=lr_dict[type(net_lst['metric']).__name__],
                                      optimizer_kwargs= {'amsgrad': True}) # NOTE this work
    
    optimizer = optim_init['optimizer']
    metric_history = {}
    with block_time() as timer:
        for i in pbar:
            optimizer.zero_grad()
            
            loss, metric_output = loss_fn(sample_dict = sample_dict, **parameter_for_registration, ) # FIXME adjust to not overlap keys in sample and registration dict
            metric_history = metric_outputs_update(start_dict= metric_history, new_data = metric_output)
            loss_history.append(-loss.item())
            loss.backward()
            optimizer.step()
            
            #============ EARLY STOPPING CHECK===========
            if early_stopping:
                
                best_loss = max(loss_history)
                if loss_history[-1] > best_loss + min_delta:
                    best_loss = loss_history[-1]
                    counter_patience = 0
                else:
                    counter_patience += 1
                if counter_patience >= patience:
                    standard_log.info(f'Early stopping at iteration {i} with best loss {best_loss}')
                    break
    execution_time = timer[0]
    net_lst['trasformation'].eval()
    net_lst['metric'].eval()

    registration_dict['affine_matrix']= net_lst['trasformation']
    registration_dict ['execution_time'] = execution_time
    registration_dict['loss_history'] = loss_history # NOTE define also ITERATION_NUMS
    # use of flatten dict to construct a explicit lr_dict for data collector
    lr_dict_flatten = get_flatten_dict(lr_dict)
    for net_lr in lr_dict_flatten.keys():
        for net in net_lst.values():
            if type(net).__name__ not in net_lr:
                continue
            else: 
                registration_dict[net_lr] = lr_dict_flatten[net_lr]
    #registration_dict['v1_lr'] = lr_dict['HomographyNet'].get('v1', 1.e-5)
    #registration_dict['vL_lr']= lr_dict['HomographyNet'].get('vL', 1.e-3)
    #registration_dict['mine_lr'] =lr_dict['MINE'].get('lr', 1.e-2)
    registration_dict['index_sampling'] = sampling

    if early_stopping:
        registration_dict['patience'] = patience
        registration_dict['best_loss']= best_loss
        registration_dict['min_delta']= min_delta
    
    return registration_dict


#                                           ==========================
#                                                RESULTS COLLECTOR
#                                           ==========================

def results_collection(
        dataloader : DataLoader,
        registration_fn : Callable[[SampleDict, dict | FrameworkConfig ], dict], 
        collector: Registration_Data_Collector,
        preprocessing_fn :Callable = general_preprocessing,
        config_dict : dict | FrameworkConfig = _fwc_dict,
        max_sample: Optional[int]= None,
        registration_name : Optional[str]=None,
        **collection_kwargs,
        )-> Registration_Data_Collector:
    '''
    foundamentally a wrapperon registration loop, to collect results althoghether

    Args:
        dataloader (DataLoader): the dataloader whose iterator is used to load data on registration loops
        registration_fn (Callable[[SampleDict, dict | FrameworkConfig ], dict]): registration/wrapper to align images, it produce a dict
        collector (Registration_Data_Collector): Data Collector to store informations in, it should be empty, but not a must
        preprocessing_fn (Callable, optional): preprocessing of the registration. Defaults to general_preprocessing.
        config_dict (dict | FrameworkConfig, optional): the framework dict of configuration as well as the main dict for wverything below. Defaults to _fwc_dict.
        max_sample (int, optional): define the dimension of the dataset to analize. Default to None
        registration_name (str, optional): the name whose collector will use for registration. None as Default
    Returns:
        Registration_Data_Collector: the updated collector with newfound datas of this registration_fn
    '''
    #1. dataloader iterator:
    data_iterator = iter(dataloader)
    registration_name = registration_name or registration_fn.__name__

    if max_sample is not None and 0 < max_sample <= len(dataloader):
        data_iterator = islice(data_iterator, max_sample)
    for sample_set in tqdm(data_iterator,total=max_sample, desc='REGISTRATION IN WORK'):
        preprocessed_sample_dict = preprocessing_fn(sample_dict=sample_set, config_dict=config_dict)
        registration_results = registration_fn(preprocessed_sample_dict, config_dict=config_dict)

        if not any('naed' in k.lower() for k in registration_results.keys()):

            coords_norm = {'image_normalization': 'coords_norm', 'name_to_use': 'naed_coords'}
            registration_results=update_with_evaluation(sample_dict=preprocessed_sample_dict, # FIXME to complete with normalization_type, use of kwargs
                                                        registration_results=registration_results,
                                                        eval_fn=wrapper_naed_drmine,
                                                        **coords_norm,
                                                        **collection_kwargs)
            
            diagonal_norm = {'image_normalization': 'diagonal', 'name_to_use': 'naed_diagonal'}
            registration_results=update_with_evaluation(sample_dict=preprocessed_sample_dict, # FIXME to complete with normalization_type, use of kwargs
                                                                    registration_results=registration_results,
                                                                    eval_fn=wrapper_naed_drmine,
                                                                    **diagonal_norm,
                                                                    **collection_kwargs)
            
        

        collector.add_registration_data(image_pair_name=preprocessed_sample_dict['sample_name'],
                                        registration_data = registration_results,
                                        registration_name = registration_name
                                        )    

    return collector
    





#                                           ==========================
#                                                GENERAL PIPELINE
#                                           ==========================


PreprocessingFn = Callable[[SampleDict | Sequence, dict], SampleDict]

RegistrationFn = Callable[[SampleDict, dict], dict]

DataCollectionFN = Callable[
    [DataLoader, # dataloader
     dict | FrameworkConfig, # config_dict
     RegistrationFn, # registrations_fn
     PreprocessingFn, # preprocessing_fn
     Registration_Data_Collector # data_collector
     ], Registration_Data_Collector

]

def run_registration_pipeline(

        registration_fn : Callable[[SampleDict, dict | FrameworkConfig ], dict], # to typize
        dataset : Optional[Dataset] = None, #optional
        dataloader : Optional[DataLoader] = None, # depending on the previous
        config_dict : dict | FrameworkConfig =_fwc_dict,
        preprocessing_fn : Callable[[SampleDict, dict| FrameworkConfig], SampleDict] = general_preprocessing,
        data_collection_fn : DataCollectionFN = results_collection,
        save_results: bool = True,
        **pipeline_kwargs
        )->Tuple[Registration_Data_Collector, Optional[Path]]:
    '''
    run the registration defined pipeline by using key point in this parametrization

    Args:
        dataset (Dataset):  the Dataset of images it get a easy access to images as path like or PIL images
        dataloader (DataLoader): dataloader starting from the dataset, previously cited
        config_dict (dict or FrameworkConfig): the dict infos containing all tipe of informations, depending also on registration and preprocessing functions, 
            otherwise only change the yaml file for configuration
            for a dict type on this level it has to have the folowing keys : 'SAVING_FORMAT' , 'RESULTS'
        preprocessing_fn (Callable[[SampleDict, dict], SampleDict]): preprocessing function, all from dataset/dataloader to input of registration_fn
        registratoion_fn (Callable): registration/wrappper to get the results in dict form
        data_collection_fn (Callable [dict, [Path, Registration_Data_Collector]]): the function to get results organized in Registration_Data_Collector
        save_results (bool): whether to save the registration results

    Required Keys
    ------------
    'SAVING_FORMAT', 'RESULTS'
    
    Returns:
        [Registration_Data_Collector, Optional[Path]]: Registration_Data_Collector is a must, if save_results, also the saving_path is delivered
    '''
    if dataloader is None and dataset is None:
        raise ValueError('one of the two must be defined, otherwise nothing to do')
    elif dataloader is None:
        dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)


    data_collector = Registration_Data_Collector()

    collected_results = data_collection_fn(
        dataloader = dataloader,
        config_dict=config_dict,
        registration_fn = registration_fn,
        preprocessing_fn = preprocessing_fn,
        collector = data_collector,
        **pipeline_kwargs
    )
    saving_path = None
    if save_results:
        saving_ks = ['SAVING_FORMAT', 'RESULTS']
        if isinstance(config_dict, dict):
            if has_required_keys(input_dict=config_dict, required_keys=saving_ks):
                saving_format = config_dict['SAVING_FORMAT']
                results_path = config_dict['RESULTS']
            else:
                saving_format, results_path = _fwc_dict.text_dict['SAVING_FORMAT'], _default_save_dir     
        else:
            saving_format = config_dict.text_dict['SAVING_FORMAT']
            results_path = config_dict.path_dict['DRIVE_RESULTS'] if 'google.colab' in sys.modules else config_dict.path_dict['RESULTS']

        saving_path = collected_results.save_data(filename=None, results_path=results_path, fmt = saving_format)

    return (collected_results, saving_path)