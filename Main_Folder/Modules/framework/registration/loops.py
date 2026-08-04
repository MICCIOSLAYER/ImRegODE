# all useful for registration loop
from Main_Folder.Modules.configuration_setting.logger_configuration import get_logger
from Main_Folder.Modules.utils import  permute_channel_layout
from Main_Folder.Modules.framework.preprocessing import normalize_img
from Main_Folder.Modules.framework.data_classes import SampleDict
from Main_Folder.Modules.configuration_setting.yaml_configuration import FrameworkConfig

import torch
import cv2
from typing import Any

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