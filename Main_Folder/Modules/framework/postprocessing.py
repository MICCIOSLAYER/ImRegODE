# all function for post processing results and preparations for evaluation
import numpy as np
import torch
from typing import Sequence

from Main_Folder.Modules.utils import get_tensor





def unnormalize_matrix(H: torch.Tensor,
                        image_shape: list[int, int],
                        transformation_map : list = []) -> torch.Tensor:
    '''
    since the homography is calculated on normalized coordinates this function unnormalize the homography matrix to be applied on the original image
    '''
    normalized_affine_matrix = H.cpu().detach().numpy()
    h, w = image_shape
    if transformation_map == []:
        transformation_map = [[2/(w-1), 0, -1], [0, 2/(h-1), -1], [0, 0, 1]] #NOTE definiscila a partire da alpha e beta così come norm...
    unnormalized_affine_matrix = np.linalg.inv(np.matmul(np.matmul(np.linalg.inv(transformation_map), normalized_affine_matrix), transformation_map))
    return torch.tensor(unnormalized_affine_matrix).to(H.device)



def decrop_matrix(H: torch.Tensor,
                    offset_x0 : int,
                    offset_y0 : int) -> torch.Tensor:
    '''
    Since the homography is calculated on cropped images,
    this function decrop the unnormalized homography matrix to be applied on the original image

    Parameters:
    H (torch.Tensor): The unnormalized homography matrix.
    offset_x0 (int): The x offset of the cropping. It is the number of pixel corresponding to the high left corner of the image, coord x.
    offset_y0 (int): The y offset of the cropping. It is the number of pixel corresponding to the high left corner of the image, coord y.

    '''
    tx= -offset_x0
    ty= -offset_y0
    transformation_map_crop_to_full = [[1, 0, tx], [0, 1, ty], [0, 0, 1]]
    
    decropped_affine_matrix = np.matmul(np.matmul(np.linalg.inv(transformation_map_crop_to_full), H), transformation_map_crop_to_full)
    return torch.tensor(decropped_affine_matrix).to(H.device)


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