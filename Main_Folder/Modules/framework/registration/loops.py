# all useful for registration loop
from Main_Folder.Modules.configuration_setting.logger_configuration import get_logger
from Main_Folder.Modules.utils import  permute_channel_layout
from Main_Folder.Modules.framework.preprocessing import normalize_img
from Main_Folder.Modules.framework.data_classes import SampleDict
from Main_Folder.Modules.framework.registration.networks import HomographyNet, MINE
from Main_Folder.Modules.configuration_setting.yaml_configuration import FrameworkConfig
import torch.nn.functional as F
import torch
from torch import nn
import numpy as np
from typing import Any, Tuple

standard_log = get_logger(__name__)

def extract_registration_params(sample_dict : SampleDict,
                        config_dict: dict |FrameworkConfig,
                        
                        device : torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        ) -> dict[str, Any]:
  '''
  it registers the images calculating the random index for the homography to compare the registration and optimize the corresponding value xy_
  it also prepare the pyramidal images permuting the index position as (H,W,C) -> (C, H, W)

  Parameters
  ------------
  sample_dict (SampleDict): the dict containing all information about the sample useful for registration loop
  config_dict (dict): the dict representing the yaml configuration file
  device (torch.device): to put the tensor on the correct device during execution and calcula

  Returns
  -------
  dict: {'I_lst': I_lst,  list[torch.Tensor] the image pyramid of the fixed image listed as torch tensor of normalized value as np.float32
  'J_lst': J_lst,  list[torch.Tensor] the image pyramid of the moving image listed as torch tensor of normalized value as np.float32
  'x_': x_, list[torch.Tensor] the x coordinates of fixed as torch tensor of normalized value as np.float64 for homography mapping
  'y_': y_, list[torch.Tensor] the y coordinates of fixed as torch tensor of normalized value as np.float64 for homography mapping
  'xy_': xy_lst,  list[torch.Tensor] the coordinates for the homography mapping
  'ind_': ind_lst,  list[torch.Tensor] the list of indeces from whose loss is calculated
  'h_': h_lst, list[int] the heights of images in pyramid, necessary due downsampling
  'w_': w_lstlist[int] the widths of images in pyramid, necessary due downsampling
  }

  '''
  
  
  # ======== GET VALS from CONFIG_DICT=============

  if isinstance(config_dict, FrameworkConfig):
    parameter_dict_configuration = config_dict.registrations['DRMINE_original']
  elif isinstance(config_dict, dict):
      parameter_dict_configuration = config_dict

  
  sampling= parameter_dict_configuration['sampling_ratio']
  L = parameter_dict_configuration['pyramid']['gaussian_levels']
  reference_pyramid= sample_dict['reference_pyramid']
  test_pyramid = sample_dict['test_pyramid']
  nChannel = sample_dict['nChannel']
  
  if not (0.0 < sampling <= 1.0) :
    standard_log.error(f' the sampling ratio must be equal up to 1.0, so {sampling} will be set to 0.1 as default')
    sampling = 0.1

  reference_lst, test_lst, heights_lst, widths_lst, xy_lst, ind_lst=[],[],[],[],[],[] #one feature for each level of the pyramid


  for s in range(L): # for each level of resolution pyramids
    ref_ = normalize_img(image=reference_pyramid[s], normalization_interval= (0,1),).to(device) # normalization for normalized coordinates
    tst_ = normalize_img(image=test_pyramid[s], normalization_interval= (0,1),).to(device) # normalization for normalized coordinates

    if nChannel>1:
        reference_lst.append(permute_channel_layout(image=ref_, target_format = 'C**')) # permute (H,W, C) -> (C, H, W)
        test_lst.append(permute_channel_layout(image=tst_, target_format = 'C**')) # permute (H,W, C) -> (C, H, W)
        height_, width_ = reference_lst[s].shape[1], reference_lst[s].shape[2]
        ind_ = torch.randperm(int(height_*width_*sampling), device=device) # FIXME usa sempre lo stesso bacino di pixel ( il primo 10%), se volessi renderlo davvero randomico dovrei usare ind_ = torch.randperm(int(h_*w_))[:int(h_*w_*sampling)] 
        ind_lst.append(ind_)
    else:
        reference_lst.append(ref_)
        test_lst.append(tst_)
        height_, width_ = reference_lst[s].shape[0], reference_lst[s].shape[1]
        ind_ = torch.randperm(int(height_*width_*sampling), device=device)# FIXME usa sempre lo stesso bacino di pixel ( il primo 10%), se volessi renderlo davvero randomico dovrei usare ind_ = torch.randperm(int(h_*w_))[:int(h_*w_*sampling)] 
        ind_lst.append(ind_)
    heights_lst.append(height_)
    widths_lst.append(width_)

    y_, x_ = torch.meshgrid([torch.arange(0,height_).float().to(device), torch.arange(0,width_).float().to(device)], indexing='ij')
    y_, x_ = 2.0*y_/(height_-1) - 1.0, 2.0*x_/(width_-1) - 1.0 # normalization [0, 0] -> [-1, -1], [h,w] -> [1, 1]
    xy_ = torch.stack([x_,y_],2)
    xy_lst.append(xy_)
  return {'reference_lst' : reference_lst,
          'test_lst': test_lst,
          'x_' : x_,
          'y_': y_,
          'xy_lst': xy_lst,
          'ind_lst': ind_lst,
          'heights_lst': heights_lst,
          'widths_lst': widths_lst
          }


#                                              =====================
#                                                TRASFORMATION FNS 
#                                              =====================


def AffineTransform(I: torch.Tensor,
                    H: HomographyNet,
                    xv: torch.Tensor, 
                    yv: torch.Tensor
                    ) -> torch.Tensor:
    '''
    Apply a homography transformation to an image.

    Args:
        I (torch.Tensor): The input image, as the image to be warped, since grid_sample has to be used I.shape should follow: (N, C, H, W)
        H (torch.Tensor): The homography matrix, to apply.
        xv (torch.Tensor): The x coordinates of the points to transform, as x vector
        yv (torch.Tensor): The y coordinates of the points to transform, as y vector

    Returns:
        torch.Tensor: The transformed image.
    '''
    convert_on_same_device = False
    
    for tensor in [I, H, xv, yv]:
        if tensor.device == 'cuda':
            convert_on_same_device=True
            
    if convert_on_same_device:
        for tensor in [I, H, xv, yv]:
            tensor.to('cuda')

    # apply homography
    xvt = (xv*H[0,0]+yv*H[0,1]+H[0,2])/(xv*H[2,0]+yv*H[2,1]+H[2,2])
    yvt = (xv*H[1,0]+yv*H[1,1]+H[1,2])/(xv*H[2,0]+yv*H[2,1]+H[2,2])
    J = F.grid_sample(I,torch.stack([xvt,yvt],2).unsqueeze(0), align_corners=True).squeeze() #NOTE the grid params in grid_sample it has to be of sahpe (N, H, W, 2)
    # NOTE use mode = 'nearest' & padding_mode='border'
    return J



#                                                ============================
#                                                          LOSS FNS
#                                                =============================

def multi_resolution_loss(I_lst : list[torch.Tensor],
                          J_lst : list[torch.Tensor],
                          xy_lst : list[torch.Tensor],
                          ind_lst : list[torch.Tensor],
                          homography_net : HomographyNet,
                          mine_net : MINE,
                          
                          
                          nChannel : int = 3,
                          
                          )-> Tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    '''
    the loss calculated along the multiresolution pyramid: loss = Sum[L]: mi/L()

    Args:
        I_lst (list[torch.Tensor]): the Multiresolution Pyramid of the REFERENCE image
        J_lst (list[torch.Tensor]): the Multiresolution Pyramid of the TEST image
        xy_lst (list[torch.Tensor]): the list of x,y coordinates on which the homography/affinity is applied (H)
        ind_lst (list[torch.Tensor]): the list of indices from which calculate the loss
        homography_net (HomographyNet): the homography matrix specified for multiresolution loss ( depending on the level s it change the compostision of exp(C))
            It has to be declared as hommography_net = HomographyNet() before the training loop and then passed as an argument to the loss function
        mine_net (MINE): nn to calculate the MI using the Donsker Varadam lower bound
        nChannel (int, optional): number of color channels. Defaults to 3.

    Returns:
        Tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]: it returns:
        loss (torch.Tensor) : the resulting summed mi calculation for all the pyramid levels ( to be backwarded in the nn training)
        z1 (list[torch.Tensor]) : the list of scores associated to the matching couple (x_i, y_i) for each pyramid level, to be used for visualization purpose
        z2 (list[torch.Tensor]) : the list of scores associated to the unmatching couple (x_i, y_j) for each pyramid level, to be used for visualization purpose
    '''
    assert I_lst[0].device == J_lst[0].device, "the two image pyramids must be on the same device"
    assert xy_lst[0].device == ind_lst[0].device, "the coordinates and the indices must be on the same device of the image pyramids"
    loss=I_lst[0].new_tensor(0.0) 
    z1, z2 = [], [] # NOTE use z1 z2 to track the loss doing on MINE net
    L = len(I_lst)
    for s in np.arange(L-1,-1,-1): #FIXME usa for s in range(L-1, -1, -1):...
      if nChannel>1: # for RGB images
          Jw_ = AffineTransform(J_lst[s].unsqueeze(0), homography_net(s), xy_lst[s][:,:,0], xy_lst[s][:,:,1]).squeeze()
          mi, zi, zj = mine_net(torch.cat([I_lst[s],Jw_],0).permute(1,2,0),ind_lst[s])
          loss = loss - (1./L)*mi
          z1.append(zi)
          z2.append(zj)
      else: # for bn images
          Jw_ = AffineTransform(J_lst[s].unsqueeze(0).unsqueeze(0), homography_net(s), xy_lst[s][:,:,0], xy_lst[s][:,:,1]).squeeze()
          mi, zi, zj = mine_net(torch.stack([I_lst[s],Jw_],2),ind_lst[s])
          loss = loss - (1./L)*mi
          z1.append(zi)
          z2.append(zj)
    
    return loss, z1, z2




def pyramid_loss(sample_dict: SampleDict,
                 trasformation_network:nn.Module,
                 metric_network: nn.Module,
                 xy_lst: list[torch.Tensor],
                 ind_lst: list[torch.Tensor],
                 **kwargs
                 )-> tuple[torch.Tensor, dict]:
    '''

    Args:
        sample_dict (SampleDict): the dict containing most informations of sample 
        trasformation_network ([nn.Module]): network responsible of transformation (warping), NOT the type-data but the instantiated object
        metric_network ([nn.Module]): network responsible of metric calculation, NOT the type-data but the instantiated object
        xy_lst list[torch.Tensor]: list referring to each resolution-level of coordinates of each pixel in image. Defaults to None.
        ind_lst list[torch.Tensor]: list referring to each resolution-level of coordinates of pixel for loss_fn in images. Defaults to None.

    Returns:
        torch.Tensor: loss calcula
    '''
    total_loss = torch.tensor(0.0)
    levels = len(sample_dict['test_pyramid'])
    test_pyramid_lst = sample_dict['test_pyramid']
    reference_pyramid_lst = sample_dict['reference_pyramid']

 
    
    for level in np.arange(levels-1, -1, -1):
        if sample_dict['nChannel']>1:
            warped_image = AffineTransform(test_pyramid_lst[level].unsqueeze(0), trasformation_network(level), xy_lst[level][:,:,0], xy_lst[level][:,:,1]).squeeze()
            level_metric_loss, zii, zij = metric_network(torch.cat([reference_pyramid_lst[level],warped_image],0).permute(1,2,0),ind_lst[level]) # FIXME substitute with metric_network(permute_channel_layout(torch.cat([reference_pyramid_lst[level],warped_image],0), target_format = "**C"),ind_lst[level])
            total_loss = total_loss - (level_metric_loss/levels)
        else:
            warped_image = AffineTransform(test_pyramid_lst[level].unsqueeze(0).unsqueeze(0), trasformation_network(level), xy_lst[level][:,:,0], xy_lst[level][:,:,1]).squeeze()
            level_metric_loss, zii, zij = metric_network(torch.stack([reference_pyramid_lst[level], warped_image], 2), ind_lst[level])
            total_loss = total_loss - (level_metric_loss/levels)

    return total_loss, {'zii': zii, 
                        'zij': zij}
