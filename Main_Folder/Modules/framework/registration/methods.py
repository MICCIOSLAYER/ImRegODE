# Useful things on registration loops

from typing import Union, Callable, Sequence, Optional, Dict, Any
from tqdm.auto import tqdm
import logging
from Main_Folder.Modules.configuration_setting.logger_configuration import get_logger
from Main_Folder.Modules.configuration_setting.yaml_configuration import FrameworkConfig
from torch import nn
import torch
import SimpleITK as sitk
from Main_Folder.Modules.utils import block_time
from Main_Folder.Modules.framework.registration.networks import HomographyNet, MINE

standard_log = get_logger(__name__)


def initialize_networks(network_classes: Optional[dict[str, type[nn.Module]]]= None,
                        network_kwargs_dict:Optional[dict]= None,
                        device : torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        )-> dict[str, nn.Module]:
    '''
    The actual wrapper to initialize networks used for trasformation and mi calculation

    Args:
        network_classes : dict[str, nn.Module] the dict containing as key the pourpose of the net, as value the net type
            as e.g. network_classes = {'trasformation' : HomographyNet, 'metric': MINE, ...}
        network_kwargs_dict : dict the dict containing as key the net type, as value the kwargs for the net
            as e.g. network_kwargs_dict = {'HomographyNet': {'device': device}, 'MINE': {'nChannel': 3}, ...}
        device : torch.device the device to use for the networks. Default is cuda if available, otherwise cpu

    Returns:
        tuple[nn.Module, nn.Module]: the couple of network responsible for the trasformation and the mutual information calculation
    '''
    # Initialize the transformation network
    model = {}
    if network_classes  is None:
        network_classes= {'trasformation': HomographyNet,
                          'metric': MINE}
    if network_kwargs_dict is None:
        network_kwargs_dict= {'HomographyNet': {}, 'MINE': {'nChannel': 3}}
        
    for pourpose, net_type in network_classes.items():
        net_kwargs = network_kwargs_dict.get(net_type.__name__, {})
        model[pourpose] = net_type(**net_kwargs).to(device)

    return model



def get_model_param_groups(model: nn.Module,
                          lr: Union[float, dict[str, float]]
                          ) -> list[dict[str, Any]]:
    """
    Get the parameter groups for a model based on the provided learning rates.

    Args:
        model (nn.Module): The model for which to get parameter groups.
        lr (Union[float, dict[str, float]]): Learning rates for the model's parameters. 
            Can be a single float for all parameters or a dictionary specifying learning rates for specific parameters.

    Returns:
        list[dict[str, Any]]: A list of parameter groups suitable for use with an optimizer.
    """


    if hasattr(model, 'get_param_groups'):
        param_groups = model.get_param_groups(lr)
        return param_groups
    else:
        if isinstance(lr, (float, int)):
            return[{'params' :model.parameters(), 'lr' : float(lr)}]
        else:
            params_dict = dict(model.named_parameters())
            param_groups = []
            for name, lr_val in lr.items():
                if name not in params_dict:
                    standard_log.error(f"Attribute {name} not found in {type(model).__name__}")
                    continue
                param_groups.append({'params': [params_dict[name]], 'lr': lr_val})
            if not param_groups:
                standard_log.critical(f'No valid parameters found in the provided lr dictionary: {list(lr.keys())}')
            return param_groups



def initialize_optimizer(
        trasformation_model : nn.Module,
        metric_model : nn.Module,
        trasformation_lr : Union[float, dict[str, float]],
        metric_lr : Union[float, dict[str, float]],
        optimizer_cls : type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[dict] = None,
        scheduler_cls : Optional[type[torch.optim.lr_scheduler.LRScheduler]] = None,
        scheduler_kwargs: Optional[dict] = None,
        )-> dict[str, Any]:
    '''
    wrapper to initialize optimizer and optionally also its scheduler

    Args:
        trasformation_model (nn.Module): the model for the trasformation
        metric_model (nn.Module): the model for the calculation of the metric (e.g., mutual information)
        trasformation_lr (Union[float, dict[str, float]]): learning rates of parameters for the trasformation model, can be a float or a dict specifying learning rates for specific parameters
        metric_lr (Union[float, dict[str, float]]): learning rates of parameters for the metric model, can be a float or a dict specifying learning rates for specific parameters
        optimizer_cls (type[torch.optim.Optimizer], optional): the optimizer class to use. Defaults to torch.optim.Adam.
        optimier_kwrags (Optional[dict], optional): additional keyword arguments for the optimizer. Defaults to None.
        scheduler_cls (Optional[type[torch.optim.lr_scheduler.LRScheduler]], optional): the learning rate scheduler class to use. Defaults to None.
        scheduler_kwargs (Optional[dict], optional): additional keyword arguments for the scheduler. Defaults to None.

    Returns:
        dict[str, Any]: the dict containing information on the scheduler and the optimizer
    '''
    optimizer_kwargs = optimizer_kwargs or {}
    scheduler_kwargs = scheduler_kwargs or {}

    params_groups = (
        get_model_param_groups(trasformation_model, 
                              trasformation_lr)
    +
        get_model_param_groups(metric_model, 
                                metric_lr)
    )


    optimizer = optimizer_cls(params_groups, **optimizer_kwargs)

    scheduler = scheduler_cls(optimizer, **scheduler_kwargs) if scheduler_cls else None

    return {'optimizer': optimizer,
            'scheduler': scheduler}



# ==================================== SITK REGISTRATION METHODS ================================

