# all function for post processing results and preparations for evaluation
import numpy as np
import torch
import SimpleITK as sitk
import sys
from typing import Sequence, Union, Protocol, Any, runtime_checkable
import matplotlib.pyplot as plt
from pathlib import Path
from Main_Folder.Modules.utils import get_tensor, get_coords_normalization_map
from Main_Folder.Modules.framework.registration.networks  import HomographyNet
from Main_Folder.Modules.framework.img_io import tensor_img_to_sitk
from Main_Folder.Modules.configuration_setting.yaml_configuration import FrameworkConfig

_fwc_dict= FrameworkConfig()
_default_save_dir = _fwc_dict.path_dict['DRIVE_RESULTS'] if 'google.colab' in sys.modules else _fwc_dict.path_dict['RESULTS'] 

#                                   ==============================
#                                    AFFINE/HOMO MATRIX OPERATIONS
#                                   ==============================


def unnormalize_matrix(H: torch.Tensor,
                        image_shape: list[int, int],
                        transformation_map : torch.Tensor = None) -> torch.Tensor:
    '''
    Denormalize the matrix using the matrix of normalization and calculating its original: 
    

    Args:
        H (torch.Tensor): the homography matrix to denormalize
        image_shape (list[int, int]): the shape of image, to get H, W in a format (H, W)
        transformation_map (torch.Tensor, optional): the normalization matrix. Defaults to None.

    Returns:
        torch.Tensor: the denormalized homography matrix
    '''
    normalized_affine_matrix = H.cpu().detach()
    h, w = image_shape
    if transformation_map is None:
        N_map = get_coords_normalization_map(height=h, width=w, data_type= H.dtype, device=H.device)
    N_inv_map = torch.linalg.inv(N_map)

    unnormalized_affine_matrix = N_inv_map @ normalized_affine_matrix @ N_map

    return torch.tensor(unnormalized_affine_matrix).to(H.device)



def decrop_matrix(H: torch.Tensor,
                    offset_y0x0: tuple[int, int]) -> torch.Tensor:
    '''
    Since the homography is calculated on cropped images,
    this function decrop the unnormalized homography matrix to be applied on the original image

    Parameters:
    H (torch.Tensor): The unnormalized homography matrix.
    offset_y0x0 (tuple[int, int]): The (y0, x0) offsets of the cropping. It is the number of pixels corresponding to the top-left corner of the cropped image.

    '''
    H = H.detach().cpu()
    ty, tx = offset_y0x0
    transformation_map_full_to_crop = torch.tensor([[1, 0, -tx], 
                                                    [0, 1, -ty], 
                                                    [0, 0, 1]], dtype=H.dtype, device=H.device)
    C = transformation_map_full_to_crop # NOTE useful for verbose pourpose

    C_inv = torch.linalg.inv(transformation_map_full_to_crop)

    decropped_affine_matrix = C_inv @ H @ C

    return decropped_affine_matrix





def get_warped_coords(test_coords:torch.Tensor |np.ndarray, 
                      affine_matrix: np.ndarray| torch.Tensor,
                      )->tuple[list[int, int]]:
    
    '''Get the warped coordinates applying the homography matrix to the test coordinates
    Parameters:
    test_coords : torch.Tensor | np.ndarray
        the coordinates to be transformed, shape (N, 2)
    affine_matrix : torch.Tensor | np.ndarray
        the homography matrix, shape (3, 3)
        
    Returns:
    tuple[list[int, int]]
        the transformed coordinates as list of integer tuples'''
    ones = torch.ones(test_coords.shape[0], 1)
    #print(f'ones is : \n{ones}  of shape {ones.shape}\n\n')
    homogeneous_points = torch.cat([test_coords, ones], dim=1)
    #print(f'homogeneous_points is : \n{homogeneous_points} of shape {homogeneous_points.shape}\n\n')
    # Apply homography
    affine_matrix = get_tensor(affine_matrix)
    transformed_points = torch.matmul(homogeneous_points, affine_matrix.T)
    #print(f'affine matrix is : \n{affine_matrix} of shape \t{affine_matrix.shape}\n\n')
    #print(f'transformed_points are : \n{transformed_points} of shape \t{transformed_points.shape}\n\n')
    # Convert back to Cartesian coordinates
    transformed_points = transformed_points[:, :2] / transformed_points[:, 2].unsqueeze(1)
    transformed_points_int_list = [[int(round(x)), int(round(y))] for x, y in transformed_points.tolist()]
    
    return transformed_points_int_list




#                   ==================== SITK PORSTPROCESSING ======================


def get_image_confrontation_SITK(image_ref : sitk.Image | torch.Tensor, #FIXME to be adjusted to wrappers, preprocessing, dataset and config_dict
                   image_test : sitk.Image | torch.Tensor,
                   outTx : sitk.Transform,
                   interpolator = sitk.sitkLinear,
                   default_pxv : int = 100,
                   save_image: bool = False,
                   folder_path: Path = _default_save_dir / 'ImgRes', 
                   name: str = 'image_sitk'                  
                   )-> sitk.Image:
    '''
    get the warped image and the fixed image from the SimpleITK images and the transformation matrix to return a visual confrontation of the differences in between

    Args:
        image_ref (sitk.Image | torch.Tensor): input image as reference
        image_test (sitk.Image | torch.Tensor): input image as the warped one
        outTx (sitk.Transform): trasformation of sitk
        interpolator (_type_, optional): interpolator to pass from discrete to conituous data. Defaults to sitk.sitkLinear.
        default_pxv (int, optional): Default pixel values for SetDefaultPixelValue method of resampler . Defaults to 100.
        visualize (bool, optional): flag to save confrontation. Defaults to False.
        folder_path (Path, optional): folder path to save image confrontation. Defaults to _default_save_dir/ 'ImgRes'.
        name (str, optional): name image. Defaults to 'A01_sitk'.

    Returns:
        sitk.Image: _description_
    '''
    IN_COLAB = 'google.colab' in sys.modules
    if isinstance(image_ref, torch.Tensor):
        image_ref = tensor_img_to_sitk(image_ref)
    if isinstance(image_test, torch.Tensor):
        image_test = tensor_img_to_sitk(image_test)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image_ref)  # Set the reference image
    resampler.SetTransform(outTx)  # Set the transformation
    resampler.SetInterpolator(interpolator)  # Set the interpolator
    resampler.SetDefaultPixelValue(default_pxv)  # Set the default pixel value

    outcome_image = resampler.Execute(image_test)  # Execute the resampling
    sigm1 = sitk.Cast(sitk.RescaleIntensity(outcome_image), sitk.sitkUInt8)
    sigm2 = sitk.Cast(sitk.RescaleIntensity(image_ref), sitk.sitkUInt8)
    image_confrontation = sitk.Compose(sigm1, sigm2, sigm1//2.0 + sigm2//2.0)  # Compose the two images


    array_image = sitk.GetArrayFromImage(image_confrontation)  # Convert to numpy array for visualization
    if array_image.shape[0] in (1, 3):  # se è un'immagine monocromatica o RGB
        array_image = np.moveaxis(array_image, 0, -1)
    plt.figure(figsize=(6, 6))
    plt.imshow(array_image)
    plt.axis('off')
    plt.title("Warped and Fixed Image")
    if save_image:
        if IN_COLAB:
            plt.savefig(folder_path/f'{name}.png', dpi=150, bbox_inches='tight')
            
        else:
            plt.savefig(folder_path / f'{name}.png')
    plt.show()
    plt.close()
    return image_confrontation



@runtime_checkable
class ElastixParameterMap(Protocol):
    def GetParameterMap(self, *args:Any,
                        **kwargs:Any,)->dict:
        ...

def get_affine_matrix_from_sitk_transform(sitk_transform: Union[sitk.AffineTransform,  ElastixParameterMap],
                                          )-> torch.Tensor:
    '''
    get the homogeneous affine matrix from the SimpleITK transform object
    as a torch.Tensor

    H = | M t_eff |
        | c   1   |
        
    where t_eff = t + c - M @ c as effective translation considering the center of rotation
    and M is the rotation matrix, t is the translation vector, c is the center of rotation
    
    Parameters:
    -----------
    
    sitk_transform : sitk.AffineTransform
        the SimpleITK affine transform object from which to extract the homogeneous affine matrix
    
    Returns:
    --------
        H_matrix : torch.Tensor
        the homogeneous affine matrix as a torch.Tensor of shape (dim+1, dim+1
    '''
    if isinstance(sitk_transform, ElastixParameterMap):
        H = np.eye(3)
        a11, a12, a21, a22, tx, ty =  sitk_transform.GetParameterMap(0)['TransformParameters']
        
        H[0, 0] = float(a11)
        H[0, 1] = float(a12)
        H[1, 0] = float(a21)
        H[1, 1] = float(a22)
        H[0, 2] = float(tx)
        H[1, 2] = float(ty)
        return torch.Tensor(H)
    dim = sitk_transform.GetDimension()
    M = np.array(sitk_transform.GetMatrix(), dtype=float).reshape((dim, dim))
    t = np.array(sitk_transform.GetTranslation(), dtype=float)
    c = np.array(sitk_transform.GetCenter(), dtype=float)  

    # compute the effective translation considering the center of rotation
    t_eff = t + c - M @ c
    # construct the homogeneous affine matrix
    H = torch.eye(dim + 1, dtype=torch.float32)
    H[:dim, :dim] = torch.Tensor(M)
    H[:dim, dim] = torch.Tensor(t_eff)


    return H


def all_on_cpu(registration_results : dict,
               
               )->dict:
    '''
    fondamentally a wrapper for data collector to put all data from gpu to cpu, eventually

    Args:
        results (dict): the result dict
        

    Returns:
        dict: the output dict where all data stored are on the cpu
    '''

    affine_matrix = registration_results.get('affine_matrix', [])
    if isinstance(affine_matrix, HomographyNet):
        registration_results['affine_matrix ']= affine_matrix(0).detach().cpu()

    for k, v in registration_results.items():
        if isinstance(v, dict):
            registration_results[k] = all_on_cpu(v)
        elif isinstance(v, torch.Tensor):
            registration_results[k]=v.detach().cpu() 

    return registration_results
