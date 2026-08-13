# use for evaluate the efficiency of certain methods of registration: as NAED and similar
# it contains also the utils for these methods
from Main_Folder.Modules.framework.data_classes import SampleDict
from Main_Folder.Modules.configuration_setting.yaml_configuration import  FrameworkConfig
from Main_Folder.Modules.configuration_setting.logger_configuration import get_logger
from typing import Optional, Union, Literal
from skimage.filters import gaussian
from skimage.color import rgb2gray
import numpy as np
import torch
from pathlib import Path

standard_log = get_logger(__name__)


def get_coords(image_name : str | SampleDict , #es A01
               ground_truth_path : Path = FrameworkConfig().path_dict['CONTROL_POINTS_FOLDER'], # in this case: control_points_[Image pair name]_1_2.txt
               )-> tuple[list[list[float]], list[list[float]]]:
    '''
    get the coordinates of control points associate to the image in list of coords format
    the format in the ground truth folder:

        [reference_point_1_x] [reference_point_1_y] [test_point_1_x] [test_point_1_y]

        [reference_point_2_x] [reference_point_2_y] [test_point_2_x] [test_point_2_y]

    it will returns the list of coords for both test(_2) and reference(_1) image in the format test(x,y) ; reference(x,y)
    '''
    if not isinstance(image_name, str):
        name = image_name['sample_name']
        image_name = name
    if not ground_truth_path.exists():
        raise FileNotFoundError(f'directory {ground_truth_path} not found')
    txt_file_path = [n for n in list(ground_truth_path.glob('*')) if n.name.__contains__(image_name)]
    reference_points, test_points = [], []
    with open(txt_file_path[0], 'r') as file:
        lines = file.readlines()
        for line in lines:
            ref_x, ref_y, test_x, test_y = line.split()
            reference_points.append([float(ref_x), float(ref_y)])
            test_points.append([float(test_x), float(test_y)])
    return test_points, reference_points


def naed_evaluation(image_couple_name: str,
                    affine_matrix: np.ndarray | torch.Tensor, 
                    image_normalization :  Literal['diagonal' , 'coords_norm'] = 'coords_norm',
                    control_points_path: Path = FrameworkConfig().path_dict['CONTROL_POINTS_FOLDER'],
                    image_dimensions: tuple | Path = (2912, 2912) ,
                    )-> float:
    '''
    Evaluate the registration using Normalized Average Euclidean Distance (NAED) metric.
    Parameters
    ----------
    image_couple_name : str
        The name of the image couple to get the control points file from the list. It has to be in the format 'A01', 'P19', 'S20', etc.
    control_points_file : Path
        The path to the control points coordinates.
    affine_matrix : np.ndarray | torch.Tensor
        The affine transformation matrix get by image registration algoritm it has to be in the complete form : | A t |
                                                                                                                | 0 a |
    image_dimensions : tuple | Path
        The dimensions of the images or the path to the image to get the dimensions to normalize.
    '''
    h,w = image_dimensions
    
    #1. get the control points for the image couple
    test_points_list, reference_points_list = get_coords(
        image_name=image_couple_name,
        ground_truth_path=control_points_path
    )

    test_points = np.array(test_points_list, dtype=np.float32)  # shape (N, D)  D = 2 
    reference_points = np.array(reference_points_list, dtype=np.float32)  # shape (N, D)

    assert test_points.shape == reference_points.shape , "Test and reference points must have the same shape"

    #2. control of affine matrix? for 2/3D or affine type? NOTE for the moment it has just to be in a quadratic form NxN
    if isinstance(affine_matrix, torch.Tensor):
        affine_matrix = affine_matrix.detach().cpu().numpy()
    affine_matrix = np.asarray(affine_matrix, dtype=np.float32)
    n, m = affine_matrix.shape
    if n != m:
        raise ValueError(f'The affine matrix must be square, got {n}x{m}')
    
    if n == 4:
        total_distance = np.nan # NOTE the 3D case has yet to be treated
        standard_log.info('the 3D case has yet to be treated')
        return total_distance
    
    elif n == 3:
        #3. for each list of coords: apply the matrix -> get the Euclidean distance
        total_distance = 0.0
        assert test_points.shape[-1] == 2, "Test points must be 2D coordinates"
        ones_column = np.ones((test_points.shape[0], 1), dtype=np.float32)  # shape (N, 1)
        homogeneous_test_points = np.hstack((test_points, ones_column))  # shape (N, 3
        homogeneous_warped_points = homogeneous_test_points @ affine_matrix.T  # shape (N, 3), already interpolated, using round/nearest neighbor
        
        incongruencies = [w for w in homogeneous_warped_points[:, 2] if w == 0]
        if len(incongruencies) > 0:
            total_distance = np.nan #raise ValueError("There are some incongruencies in this transformation, some points have w=0") #FIXME return NaN o gestisci in altro modo
            return total_distance
        
        else: 
            warped_points = np.rint([[x_w/w, y_w/w] for (x_w, y_w, w) in homogeneous_warped_points]).astype(np.int32)  # shape (N, 2)
            total_distance += np.linalg.norm(warped_points - reference_points, axis=1).mean()

    #4. normalize the NAED value
    if image_normalization == 'diagonal':
        #4.1 get the NAED value using image diagonal to normalize
        ratio = np.sqrt(h**2 + w**2)
    elif image_normalization == 'coords_norm':
        #4.2 get the NAED value using the norm of the image dimensions to normalize
        ratio = h
        
    #5. return NAED value in float
    naed_value = total_distance / ratio
    
    return naed_value



def mattes_mutual_information(image1, image2, bins=32):
    """
    Calculate Mattes Mutual Information (MMI) between two images.

    Parameters:
    -----------
    image1 : np.ndarray
        The first image (grayscale or RGB).
    image2 : np.ndarray
        The second image (grayscale or RGB).
    bins : int
        Number of bins for the histogram.

    Returns:
    --------
    float
        The Mattes Mutual Information value.
    """
    # Convert images to grayscale if they are RGB
    if image1.ndim == 3:
        image1 = rgb2gray(image1)
    if image2.ndim == 3:
        image2 = rgb2gray(image2)

    # Apply Gaussian smoothing
    image1 = gaussian(image1, sigma=1)
    image2 = gaussian(image2, sigma=1)

    # Compute joint histogram
    hist_2d, _, _ = np.histogram2d(image1.ravel(), image2.ravel(), bins=bins)

    # Normalize the histogram
    pxy = hist_2d / float(np.sum(hist_2d))

    # Compute marginal probabilities
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    # Compute mutual information
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    mmi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

    return mmi

def joint_histogram_mutual_information(image1, image2):
    # Compute the joint histogram
    hist, x_edges, y_edges = np.histogram2d(image1.ravel(), image2.ravel(), bins=100)
    pxy = hist / float(np.sum(hist))
    
    # Compute marginal probabilities
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0

    # Compute the joint entropy
    joint_entropy = -np.sum(pxy[nzs] * np.log(pxy[nzs]))
    
    # NOTE suggerito da copilot
    # Compute the marginal entropies 
    hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
    hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
    
    # Compute mutual information
    mi = hx + hy - joint_entropy
    #return mi

    # NOTE dal notebook:
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs])) # this is the original one
    