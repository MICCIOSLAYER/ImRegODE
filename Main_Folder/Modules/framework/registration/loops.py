# all functions to be used inside the registration loop
from Main_Folder.Modules.configuration_setting.logger_configuration import get_logger
from Main_Folder.Modules.utils import  permute_channel_layout, tensor_img_rgb2bn, get_tensor
from Main_Folder.Modules.framework.preprocessing import normalize_img
from Main_Folder.Modules.framework.postprocessing import get_warped_coords
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

  
    

  if isinstance(config_dict, dict):
      parameter_dict_configuration = config_dict
  else:
      parameter_dict_configuration = config_dict.registrations['DRMINE_original']
  

  
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

# loss on points


def loss_reference_points(reference_list : list[tuple[int, int]], # for I image
                          homography_matrix : torch.Tensor,
                          test_list : list[tuple[int, int]], # for J image
                          ) -> float :
    ''' given two list of coordinates, it get the euclidean distance between these two:
    calculated as the difference between the test point and the affine transformation of the coordinate through the homography tensor
    Args:
        reference_list (list[[int, int]]): list of coordinate points of the image of reference
        test_list (list[[int, int]]): list of coordinate points of the image to test the homography on
        homography_matrix (torch.Tensor): homography matrix to apply to the reference points get by the homography_net/ loss 
    '''
    if len(reference_list) != len(test_list):
        raise ValueError("The two lists must have the same length")
    else:
        loss = 0.0
        test_coords = get_warped_coords(torch.Tensor(test_list), homography_matrix)
        for i in range(len(reference_list)):
            x_ref, y_ref = reference_list[i]
            x_test, y_test = test_coords[i]
            loss += np.sqrt((x_ref - x_test)**2 + (y_ref - y_test)**2)
        return loss



def loss_value_points(reference_points : list[tuple[int, int]],  
                          test_points : list[tuple[int, int]], 
                          reference_image : torch.Tensor, 
                          warped_image : torch.Tensor,
                          homography_matrix : torch.Tensor,
                          ) -> float :
    ''' 
    given two list of coordinates(reference_points), 
    it get the differences in values between the test image and the warped one in these coordinates

    Assume that all the images both the fixed and the warped one are both of same shape and resulted from the dataloader,
    ready to be processed by the net of shape [3, N ,N] or [N ,N]

    Args:
        reference_points (list[[int, int]]): list of coordinate points of the image of reference I

        test_points (list[[int, int]]): list of coordinate points of the image to test the homography on J

        homography_matrix (torch.Tensor): homography matrix to apply to the reference points get by the homography_net/ loss H

        reference_image the image of reference I

        warped_image the image to be tested J
    '''
    if len(reference_points) != len(test_points):
        raise ValueError("The two lists must have the same length")
    else:
        loss = 0.
        test_points = torch.Tensor(test_points,
                                   device = homography_matrix.device,                                   
                                   )
        test_coords = get_warped_coords(test_points, homography_matrix)

        for test_coord, reference_coord in zip(test_points, reference_points):
            x_ref, y_ref = reference_coord 
            x_test, y_test = test_coord

            if reference_image.squeeze().ndim == 3: # nChannel==3
                reference_image= permute_channel_layout(image=reference_image, target_format= 'C**')
                warped_image= permute_channel_layout(image=warped_image, target_format= 'C**')
                reference_value = torch.sum(reference_image[:, y_ref, x_ref]).item()
                warped_value = torch.sum(warped_image[:, y_test, x_test]).item() # NOTE control for the ordere of x,y

            elif reference_image.squeeze().ndim == 2:
                reference_value = reference_image[x_ref, y_ref].item()
                warped_value = warped_image[x_test, y_test].item()
            loss += np.abs(reference_value - warped_value)
        return loss


# loss on full images

def loss_value_images(I : torch.Tensor,
                      J : torch.Tensor,
                      normalization : bool = True,
                      )-> float:
    ''' get the loss value between two images, normalizing if requested on N pixel
    
    Parameters
    ----------
    I, J (torch.Tensor): respecting the reference and the tested or warped image
    normalization (bool): the flag to normalize on dimensions of images'''
    if I.shape != J.shape:
        raise ValueError("the two images must have the same shape")
    else:

        I_norm = normalize_img(I.squeeze())
        J_norm = normalize_img(J.squeeze())
        I_norm = tensor_img_rgb2bn(permute_channel_layout(image=I_norm, target_format='C**'))
        J_norm = tensor_img_rgb2bn(permute_channel_layout(image=J_norm, target_format='C**'))
        if normalization:
            shape_normalizator = I.numel()
            return torch.abs(I_norm - J_norm).sum()/shape_normalizator
        else:
            return torch.abs(I_norm - J_norm).sum()



def mse_loss(img_fixed: np.ndarray | torch.Tensor,
             img_moving: np.ndarray |torch.Tensor,
             )-> torch.Tensor:
    ''' get the mean squared error loss between two images
    
    Parameter:
    img_fixed (np.ndarray | torch.Tensor): the fixed image
    img_moving (np.ndarray | torch.Tensor): the moving image
    
    Returns:
    float: the mean squared error loss between the two images, normalized over their size
    '''

    if img_fixed.shape != img_moving.shape:
        raise ValueError("the two images must have the same shape")
    img_fixed, img_moving = get_tensor(img_fixed), get_tensor(img_moving)
    img_fixed = permute_channel_layout(get_tensor(img_fixed).squeeze(), target_format='**C')
    img_moving = permute_channel_layout(get_tensor(img_moving).squeeze(), target_format='**C')

    return torch.sum((img_fixed - img_moving)**2)/img_fixed.numel()


