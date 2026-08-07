# script to manage and to categorize all the input/output functions
from typing import Union, Optional
import SimpleITK as sitk
from Main_Folder.Modules.configuration_setting.logger_configuration import get_logger
from Main_Folder.Modules.utils import tensor_img_rgb2bn, permute_channel_layout
import torch
from airlab.utils.image import Image as AirlabImage

import os
import py7zr
import requests
import zipfile
import numpy as np
from pathlib import Path

standard_log = get_logger(__name__)

def get_dataset(
        dataset_url: str,
        destination_path: Path,
        dataset_name: str = '',
        remove_zip: bool = False
) -> None:
    """
    Download a dataset from a given url and decompress it in a given destination path.

    Parameters
    ----------
    dataset_url : str
        The url of the dataset to download.
    destination_path: Path
        The path where the dataset will be decompressed.
    dataset_name : str, optional
        The name of the dataset file to download.
    remove_zip : bool, optional
        Whether to remove the zip file after decompression.
    """
    if dataset_name == '':
        dataset_name = Path(dataset_url).name
    
    

    # Create the destination folder if it does not exist
    if destination_path.exists():
        standard_log.info(f'The {destination_path} folder already exists')
    else:
        os.makedirs(destination_path, exist_ok=True)

    # Control if the dataset_name is present in the folders list of destination_path
    dataset_file = destination_path / dataset_name

    for _, folders, _ in os.walk(destination_path): # FIXME to adjust, control better
        for fold in folders:
            if dataset_name.upper() == fold.upper():
                standard_log.info(f'The {dataset_name} dataset is already extracted in the {destination_path} folder\nNo need to extract it again')
                return None

    # Download and save the dataset from the given url
    if dataset_file.exists():
        standard_log.info(f'{dataset_name} already exists, no need to download it again')
    else:
        zip_request = requests.get(dataset_url)
        with open(dataset_file, 'wb') as f:
            f.write(zip_request.content)

    # Decompress the dataset if not already done
    if dataset_file.suffix == '.7z':
        with py7zr.SevenZipFile(dataset_file, mode='r') as archive:
            archive.extractall(path=destination_path)
    elif dataset_file.suffix == '.zip':
        with zipfile.ZipFile(dataset_file, 'r') as archive:
            archive.extractall(path=destination_path)
    else:
        standard_log.error(f'Unsupported archive format: {dataset_file.suffix}')
        return None

    if remove_zip:
        os.remove(dataset_file)

    standard_log.info(f'{dataset_name} has been successfully downloaded and extracted to {destination_path}')
    return None

#                                      =============     AIRLAB    ==================

def airlab_read_image(image: Union[Path, torch.Tensor], config_dict:dict)-> AirlabImage:
    '''
    Read an image producing the airlabImage type

    Args:
        image (Union[Path, torch.Tensor]): input image from dataset in Path or tensor type format
        config_dict (dict): configuration dict from yaml file

    Returns:
        AirlabImage: _description_
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(image, Path):
        image_airlab= AirlabImage.read(image, dtype=torch.float32, device=device)
    else: # if isinstance(fixed_image, torch.Tensor) or isinstance(fixed_image, np.ndarray): # tensor type (1, 1, H, W) or (1, C, H, W)
        ref_image=tensor_img_rgb2bn(image, normalization_type=config_dict['constant']['text-like']['img_normalization'])
        sitk_input = ref_image.cpu() if torch.is_tensor(ref_image) and ref_image.is_cuda else ref_image
        image_airlab = AirlabImage(
            tensor_img_to_sitk(sitk_input),
            torch.float32,
            device)
    return image_airlab

#                                =============         SimpleITK             ==================

def tensor_img_to_sitk(img : torch.Tensor | np.ndarray
                       )-> sitk.Image:
    '''
    convert a torch tensor image to a SimpleITK image

    Parameter
    ---------
    img: the input image to be converted (torch.Tensor or np.ndarray)
    '''
    if isinstance(img, torch.Tensor):
        img = permute_channel_layout(image=img, target_format='**C').squeeze().detach().cpu().numpy()

    if img.ndim == 2: 
        return sitk.GetImageFromArray(img.astype(np.float32))
    
    if isinstance(img, np.ndarray) and img.shape[0] == 3: # multi-channel image
       img = np.transpose(img, (1, 2, 0)) # change to (H, W, C) format

    rgb_to_gray_sensitivity = [0.2898, 0.5870, 0.1140] # RGB to gray sensitivity
    img = np.dot(img, rgb_to_gray_sensitivity) # convert to grayscale

    return sitk.GetImageFromArray(img.astype(np.float32)) # convert to SimpleITK image