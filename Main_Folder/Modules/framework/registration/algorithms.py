# all registration algorithms, it contains the complete wraps of registration, 
# final products registration that give out result to be collected
from typing import Sequence, Callable, Optional, Union, Any, Tuple
from Main_Folder.Modules.framework.registration.methods import initialize_networks, initialize_optimizer
import Main_Folder.Modules.framework.registration.networks 
from Main_Folder.Modules.utils import block_time, concatenate_paths
from Main_Folder.Modules.framework.metrics import metric_outputs_update
from Main_Folder.Modules.framework.registration.loops import multi_resolution_loss, drmine_registration_loop
from Main_Folder.Modules.framework.data_classes import SampleDict, Registration_Data_Collector
from Main_Folder.Modules.configuration_setting.yaml_configuration import FrameworkConfig
from Main_Folder.Modules.framework.postprocessing import get_affine_matrix_from_sitk_transform
from Main_Folder.Modules.framework.preprocessing import general_preprocessing, airlab_mask_configuration
from Main_Folder.Modules.configuration_setting.logger_configuration import  get_logger
from Main_Folder.Modules.framework.img_io  import tensor_img_to_sitk, airlab_read_image
from Main_Folder.Modules.framework.evaluations import naed_evaluation
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


standard_log = get_logger(__name__)





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
    
    Returns
    -------
    outTx : sitk.Transform
        The resulting transformation to easly access the trasformation through sitk methods
    registration : sitk.ImageRegistrationMethod
        The registration object containing all the registration parameters and methods
    registration_time_taken : float
        The time taken for the registration process.
    '''

    if isinstance(config_dict, dict):
        mmi_sitk_dict = config_dict
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
                     config_dict:dict,)-> tuple: # NOTE see the output
    '''
    Mattes Mutual Information (MMI) between two images using SimpleITK.
    
    Parameters
    ----------
    reference_image : np.ndarray | torch.Tensor
        The fixed image.
    test_image : np.ndarray | torch.Tensor
        The warped image.

    
    Returns
    -------
    float
        The Normalized Mutual Information value.
    '''
    ncc_centred_dict = config_dict['registrations']['simpleITK_original']['NCC_centred']
    
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
              config_dict:dict,
              )->tuple[Any, sitk.ImageRegistrationMethod, float]: # NOTE see the output
    '''
    Mattes Mutual Information (MMI) between two images using SimpleITK.
    
    Parameters
    ----------
    reference_image : np.ndarray | torch.Tensor
        The reference image.
    test_image : np.ndarray | torch.Tensor
        The test image.

    
    Returns
    -------
    float
        The Normalized Mutual Information value.
    '''


    # normalization & Gaussian smoothing:
    #reference_image = sitk.DiscreteGaussian(sitk.Normalize(reference_image), 2.0) #removed since not present in the article
    #test_image = sitk.DiscreteGaussian(sitk.Normalize(test_image), 2.0) #removed since not present in the article
    jhmi_sitk_dict= config_dict['registrations']['simpleITK_original']['JHMI']
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
              config_dict : dict,
              )->tuple[Any, sitk.ImageRegistrationMethod, float]: # NOTE see the output
    '''
    Mean Squared Error (MMI) between two images using SimpleITK.
    
    Parameters
    ----------
    fixed_image : np.ndarray | torch.Tensor
        The fixed image.
    warped_image : np.ndarray | torch.Tensor
        The warped image.

    
    Returns
    -------
    float
        The Normalized Mutual Information value.
    '''
    mse_sitk_dict = config_dict['registrations']['simpleITK_original']['MSE']
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
             config_dict: dict,
           )-> tuple[Any, sitk.ImageRegistrationMethod, float]: # NOTE see the output
    '''
    Registration of two images using Normalized Cross Correlation (NCC) of SimpleITK as metric.
    
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
    ncc_sitk_dict = config_dict['registrations']['simpleITK_original']['NCC']
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



#                                    =================================
#                                            AIRLAB REGISTRATION
#                                    =================================

#-------------------------------------------WRAPPER-------------------------------------------

def airlab_wrapprer_for_registration(sample_dict:SampleDict,
                                     config_dict:dict | FrameworkConfig,
                                     airlab_registration_fn: Callable[[AirlabImage, AirlabImage, dict], dict],
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

    Returns:
        dict: registration results of airlab_registration_fn
    '''
    if isinstance(config_dict, dict):
        airlab_config_dict = config_dict
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
        
        affine_matrix, registration_data = airlab_registration_fn(reference_image=airlab_reference_image,
                                                                                reference_mask = airlab_reference_mask,
                                                                                test_mask=airlab_test_mask,
                                                                                moved_image=airlab_test_image,
                                                                                config_dict=config_dict,
                                                                              )
    else:
        affine_matrix, registration_data = airlab_registration_fn(reference_image=airlab_reference_image,
                                                                                moved_image=airlab_test_image,
                                                                                config_dict=config_dict,
                                                                                )
    image_name_couple = sample_dict['sample_name']
    naed_diagonal = naed_evaluation(image_couple_name=image_name_couple,
                                affine_matrix=affine_matrix,
                                image_normalization='diagonal')
    naed_norm_coord = naed_evaluation(image_couple_name=image_name_couple,
                                affine_matrix=affine_matrix,
                                image_normalization='coords_norm')
    
    registration_data['naed_diagonal'] = naed_diagonal
    registration_data['naed_coords'] = naed_norm_coord
    registration_data['affine_matrix'] = affine_matrix.detach().cpu().numpy()



    
    return registration_data



#                                    =================================
#                                            DRMINE REGISTRATION
#                                    =================================

#-------------------------------------------WRAPPER-------------------------------------------

def wrapper_drmine_registration_loop(
        
        parameter_for_registration:dict,
        sample_dict: SampleDict,
        config_dict: dict| FrameworkConfig,
        model_nets : Optional[dict]= None,
        
        loss_fn : Callable[..., Union[float, Sequence[float]]]=multi_resolution_loss,
        network_classes : Optional[dict[str, type[nn.Module]]] = {},
        network_initialization_params : dict = {},
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
        parameter_for_registration (dict): parameter useful for registration 
        sample_dict (SampleDict): input point of data, complete and ordered in a typedDict datatype
        config_dict (dict): dictionary of configuration, if different from the yaml configuaration obj: it has to have the following keys:
            'lr', 'n_iterations', 'sampling_ratio', 'patience', 'min_delta'; 'lr' has to be a dict with the following structure:{'MINE': 0.01, 'HomographyNet': {'vL': 0.001, 'v1': 1e-05}}
        drmine_registration_fn (Callable, optional): registration loop fn. Defaults to drmine_registration_loop.
        progression_bar (bool, optional): option to visualize the progression bar. Defaults to True.
        device (torch.device, optional): _description_. Defaults to torch.device('cuda' if torch.cuda.is_available() else 'cpu').

    Returns:
        dict: dict of results from registration and some other measurments
    '''

    if isinstance( config_dict, dict):
        drmine_registration_dict = config_dict
    elif isinstance(config_dict, FrameworkConfig):
        drmine_registration_dict = config_dict.registrations['DRMINE_original']
    
    
    lr_dict = drmine_registration_dict['lr']
    iterations= drmine_registration_dict['n_iterations']
    sampling= drmine_registration_dict['sampling_ratio']
    patience= drmine_registration_dict['patience']
    min_delta = drmine_registration_dict['min_delta']

    network_initialization_params['MINE']['nChannel'] = sample_dict['nChannel']
        
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


    #=========== GET THE USEFUL PARAMS FOR THE TRAINING LOOP================== NOTE add information to parameter for registration to use it as registration dict
    # FIXME adjust fns to not use of the assigning vals to variables
    I_lst = parameter_for_registration['reference_lst']
    J_lst = parameter_for_registration['test_lst']
    xy_lst = parameter_for_registration['xy_lst']
    ind_lst = parameter_for_registration['ind_lst']

    parameter_for_registration['trasformation_network'] = net_lst['trasformation']
    parameter_for_registration['metric_network']= net_lst['metric']
    
    # NOTE in case of other type of loss_fn add here in parameter_for_registration


    optim_init =  initialize_optimizer(trasformation_model=net_lst['trasformation'],
                                      metric_model = net_lst['metric'],
                                      trasformation_kwargs=lr_dict[type(net_lst['trasformation']).__name__], # NOTE net_lst contain objects already instantiated, so type(..) is compulosry
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
    registration_dict['v1_lr'] = lr_dict['HomographyNet'].get('v1', 1.e-5)
    registration_dict['vL_lr']= lr_dict['HomographyNet'].get('vL', 1.e-3)
    registration_dict['mine_lr'] =lr_dict['MINE'].get('lr', 1.e-2)
    registration_dict['index_sampling'] = sampling
    if early_stopping:
        registration_dict['patience'] = patience
        registration_dict['best_loss']= best_loss
        registration_dict['min_delta']= min_delta
    
    return registration_dict


#                                           ==========================
#                                                GENERAL PIPELINE
#                                           ==========================


PreprocessingFn = Callable[[SampleDict | Sequence, dict], SampleDict]
RegistrationFn = Callable[[SampleDict, dict], dict]


def run_registration_pipeline(dataset : Dataset,                              
                              registration_fn: RegistrationFn,
                              config_dict : dict | FrameworkConfig,
                              preprocessing_fn: Optional[PreprocessingFn] = general_preprocessing,
                              saving_path: Path = None,
                              
                              )-> Tuple[Registration_Data_Collector, Optional[Path]]:
    '''
    It RUN a registration Pipeline to actually get the registration results to be put in the DataCollector created
    the PIPELINE: get the dataset as first input:
    DATASET ----> PREPROCESSING ---> REGISTRATION to get results

    Args:
        dataset (Dataset): the Dataset of images it get a easy access to images as path like or PIL images
        registration_fn (RegistrationFn): main focus of the work, analyze, process, calculate informations about the image pair returning a defined formatted results
            to be collected in the designated object
        config_dict: the dict infos containing all tipe of informations, depending also on registration and preprocessing functions, 
            otherwise only change the yaml file for configuration
            for a dict type on this level it has to have the folowing keys : 'SAVING_FORMAT' , 'RESULTS', 'batch_size', 'max_sample'
        preprocessing_fn (PreprocessingFn): adjust the images as requested for the Pipeline and the experiment

        save_results (bool): whether to save the registration results
        project_root (Path): the root path of the project to save results in the right folder, if used from script use Path(__file__) as start_path

    Returns:
        Registration_Data_collector: The collector of results
    '''
    # 0. PRESET ALL THE PARAMETERS OR CONSTANT FROM YAML OR MANUAL
    
    if isinstance(config_dict, dict):
        registration_dict = config_dict
        saving_format = config_dict['SAVING_FORMAT']
        results_path = config_dict['RESULTS']
        
    else:
        registration_dict = config_dict.num_dict
        saving_format = config_dict.text_dict['SAVING_FORMAT']
        results_path = config_dict.path_dict['RESULTS']

    data_collector = Registration_Data_Collector()
    
    batch_size = registration_dict['batch_size'] 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dataset_iterator = iter(dataloader)
    max_sample = registration_dict['max_sample'] 
    
    #1. DEFINE THE PIPELINE
    for sample in tqdm(islice(dataset_iterator, max_sample), total=max_sample, desc="Running Registration Pipeline"):
        
        preprocessed_sample = preprocessing_fn(sample, config_dict)
        registration_results = registration_fn(preprocessed_sample)

        data_collector.add_registration_data(registration_results)
    
    #2. SAVE RESULTS
    if saving_path and saving_path.exists():
        save_results_in = saving_path
    else:
        save_results_in = results_path
        result_type = saving_format
        
        path_to_results = data_collector.save_data(filename='wrapper_trial', fmt=result_type, results_path=save_results_in) # FIXME use config_dict

    
    

    return (data_collector, path_to_results)