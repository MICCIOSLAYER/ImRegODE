# all registration algorithms, it contains the complete wraps of registration, 
# final products registration that give out result to be collected
from typing import Sequence, Callable, Optional, Union, Any
from Main_Folder.Modules.framework.registration.methods import initialize_networks, initialize_optimizer
import Main_Folder.Modules.framework.registration.networks 
from Main_Folder.Modules.utils import block_time
from Main_Folder.Modules.framework.metrics import metric_outputs_update
from Main_Folder.Modules.framework.registration.loops import multi_resolution_loss, drmine_registration_loop
from Main_Folder.Modules.framework.data_classes import SampleDict, FrameworkConfig
from Main_Folder.Modules.framework.postprocessing import get_affine_matrix_from_sitk_transform
from Main_Folder.Modules.configuration_setting.logger_configuration import  get_logger
from Main_Folder.Modules.framework.img_io  import tensor_img_to_sitk
from Main_Folder.Modules.framework.evaluations import naed_evaluation
import SimpleITK as sitk
from pathlib import Path
from tqdm.auto import tqdm
from torch import nn
import torch
import numpy as np


standard_log = get_logger(__name__)





#                                    =================================
#                                            SITK REGISTRATION
#                                    =================================
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


#                                    =================================
#                                           ELASTIX REGISTRATION
#                                    =================================



#                                    =================================
#                                            AIRLAB REGISTRATION
#                                    =================================



#                                    =================================
#                                            DRMINE REGISTRATION
#                                    =================================
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
