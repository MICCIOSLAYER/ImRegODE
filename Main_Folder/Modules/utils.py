# FIle containing function f trasversal utility
import sys
import os
from pathlib import Path
#import numpy as np
#import yaml
from typing import  Literal, Optional, Any, Union
import numpy as np
import SimpleITK as sitk
from collections.abc import Mapping
import torch
from torch import nn
from Main_Folder.Modules.configuration_setting.logger_configuration import get_logger
from Main_Folder.Modules.configuration_setting.yaml_configuration import FrameworkConfig
from timeit import default_timer as timer
from contextlib import contextmanager
from functools import wraps
import time
from datetime import datetime


standard_log = get_logger(__name__)

# ======================================================================================
#                                     TIMING UTILITIES
# ======================================================================================


# creation of a time tracker for functions as decorator:
def timeit(func):
    '''it get the time taken by a function to execute, to get the value
    of time from the function results, use it as follows: 

    @timeit
    def my_function(...):
        ...
        return result

    result, time_taken = my_function(...)'''
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        time_taken = end - start
        # NOTE use logging to get the time taken info
        return result, time_taken
    return wrapper



#creation of a context manager to get the time taken by a code block:
@contextmanager
def block_time():
    '''Context manager to measure the time taken by a code block.
    Usage: 
    with block_time() as t:
        # code block to measure
        ...
    time_taken = t[0]
    '''
    start = time.perf_counter()
    t = [None]
    yield t
    t[0] = time.perf_counter() - start
    
    
#
def get_data_time()->str:
    '''Get the current date and time as a formatted string specifically for file naming.
    
    Returns
    -----
    the str as format of YearMonthDay-HourMinuteSecond'''
    now = datetime.now()
    current_time = now.strftime("%Y%m%d-%H%M%S")
    return current_time


# ======================================================================================
#                                     PATHS UTILITIES
# ======================================================================================


def get_root_path(
    start_path: Path | None = None,
    project_name: str = "ImRegODE"
) -> Path:
    '''
    Help to get the project directory, if used inside a ipynb use Path.cwd(), None otherwise is good

    Args:
        start_path (Path | None, optional): The starting path for the search. Defaults to None.
        project_name (str, optional): the name of project directory. Defaults to "ImRegODE".

    Raises:
        FileNotFoundError: There no parent directory named as project_name, maybe it's not the right project or the project is not in the parent directories

    Returns:
        Path: The path to the project directory.
    '''

    if start_path is None:
        start_path = Path(__file__).resolve()

    start_path = start_path.resolve()

    for path in [start_path] + list(start_path.parents):

        if path.stem == project_name:
            standard_log.debug(f'the root path is : {path}')
            return path
    
    raise FileNotFoundError(
        f"Project root '{project_name}' not found." 
    )


def concatenate_paths(root: Path, relative_path: Path | str) -> Path:
    '''
    Concatenate root path with a relative path and return the full path.
    Args:
        root (Path): The root directory path.
        relative_path (str): The relative path to concatenate with the root.
    Returns:
        Path: The full concatenated path.
    '''
    # 1. Get the path of the files, assert the existence of root dir and the absolute( of relative path)
    
    if not root.exists():
        standard_log.error(f'Root path {root} does not exist. Please check the path and try again.')
        raise FileNotFoundError(f'Root path {root} does not exist.')
    if isinstance(relative_path, str): # usa config_file
        relative_path = Path(relative_path)
        if not isinstance(relative_path,Path):
            try:
                relative_path = Path(*relative_path)
                if not isinstance(relative_path,Path):
                    logs.error(f'Error converting relative_path to Path: {relative_path}')
            except Exception as e:
                raise TypeError(f'Error converting relative_path to Path, since its type is: {type(relative_path)}.')

    # 2. define the different dirs of both path in two different Sequence: lists/tuples
    
    # 3. Easy Case: the concatenation (root/relative_path) exists and is correct, return it
    if (root / relative_path).exists():
        return root / relative_path
    
    # 4. Not that simple case: the concatenation doesn't exist, remove repetition: start from the root, 
    # then remove every dir in the relative path that is already in the root and then concatenate the rest of the relative path to the root, 
    # return the full path

    root_parts = root.parts
    rel_parts = relative_path.parts

    # Trova massimo overlap ordinato:
    # suffix di root == prefix di relative_path
    max_overlap = min(len(root_parts), len(rel_parts))
    overlap = 0

    for k in range(max_overlap, 0, -1):
        if root_parts[-k:] == rel_parts[:k]:
            overlap = k
            break

    merged_path = root.joinpath(*rel_parts[overlap:])

    return merged_path
    

def walk_through_dir(dir_path):
  """
  Walks through dir_path returning dimension of itscontents.
  Args:
    dir_path (str or pathlib.Path): target directory
  
  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
      in the format:
        for dirpath, dirnames, filenames in os.walk(dir_path):
          print(f"There are {len(dirnames)} directories and {len(filenames)} files in '{dirpath}'.")
 
  """
  
  for dirpath, dirnames, filenames in os.walk(dir_path):
    standard_log.info(f"There are {len(dirnames)} directories and {len(filenames)} files in '{dirpath}'.")



def get_dirs_of(file_path : Path,
                )->list[str]:
    '''
    given a certain path it get all the dir as a list of str to get this image
    in the format of a list as:

    [test/reference, deformation_type, file_name]

    ['Reference', 'Longitudinal_Studies', 'A01_1.jpg']
    '''
    if not file_path.is_file():
        image_type, image_deformation = file_path.parts[-2:]
        return [image_type, image_deformation]
        
    image_type, image_deformation, file_name =  file_path.parts[-3:]
    return [image_type, image_deformation, file_name]

Saving_Type = Union[Path, str]
ExtensionType = Literal['png', 'pkl', 'txt', 'csv'] # FIXME to be completed

def get_saving_name(format : ExtensionType,
                    save_path: Saving_Type = FrameworkConfig().path_dict['RESULTS'],
                    saving_name : str = '',
                    overwrite : bool = False,
                    )->Path:
    '''
    easy way to set the name of file to save

    Args:
        save_path (Saving_Type): saving path of the file
        saving_name (str, optional): the saving name of the object. Defaults to ''.
        overwrite (bool, optional): flag to define if a change in saving_name is necessary. Defaults to False.

    Returns:
        Path: the path of the saving file or dir 
    '''
    date_time_name = get_data_time()
    save_path = Path(save_path)
    extension = f'.{format.lower()}'
   

    if save_path.is_dir() and saving_name:
        return_path= Path(save_path) 
        saving_name = Path(saving_name).name
    elif save_path.is_dir() and not saving_name:
        return Path(save_path) / Path(f'{date_time_name}').with_suffix(f'{extension}')
    elif not save_path.is_dir():
        if save_path.parent.exists() and save_path.parent != Path('.'):
            return_path = save_path.parent
            saving_name = save_path.name            
        elif save_path.parent == Path('.'): # NOTE usare else: ? 
            return_path = FrameworkConfig().path_dict['RESULTS'] 
            saving_name = save_path.name
    
    saving_file_path = Path(return_path) / Path(saving_name).with_suffix(f"{extension}")
    if not overwrite and  saving_file_path.exists():
        saving_name = f'{date_time_name}_{saving_name}'

    return Path(return_path)/Path(saving_name).with_suffix(f"{extension}")
             


# ======================================================================================
#                                     OBJS UTILITIES(maintaining their nature)
# ======================================================================================


def deep_update(
    base_dict: dict,
    higher_priority_dict: dict,
) -> dict:
    '''
    A recursive function to update a dict starting from a base: BASE_DICT and using the HIGHER_PRIORITY_DICT to update/add k,v to the base dict


    Args:
        base_dict (dict): The base dictionary to be updated.
        higher_priority_dict (dict): The dictionary with higher priority values that will be used to update the base dictionary.

    Returns:
        dict: The updated dictionary.
    '''

    merged = base_dict.copy()

    for key, value in higher_priority_dict.items():

        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_update(
                merged[key],
                value,
            )

        else:
            merged[key] = value

    return merged


def get_flatten_dict(complex_dict : dict,
                 separator: str =  '.',
                 final_key : Optional[str] = None
                 )->dict[str, Any]:
    '''
    A function to flatten complex dict: 
    Eg. complex_dict = {'first_layer': {'second_layer' : {'a': 1, 'b': 0}}}
    

    Args:
        complex_dict (dict): the starting dict where some vals are dict themselves
        separator (str. Default '.'): separator to use for define the final_key

    Returns:
        dict[str, Any]: flatetn dict. no nidificated dict:
        Eg. flatten_dict = {'first_layer.second_layer.a' : 1, 'first_layer.second_layer.b': 0}
    '''
    
    flatten_dict : dict[str, Any] = {}
    
    for k, v in complex_dict.items():
        if final_key is None:
            current_key = str(k)
        else:
            current_key = str(final_key) + separator + str(k)
        
        if  not isinstance(v, dict):
            flatten_dict[current_key] = v
        
        else:
            flatten_dict.update(get_flatten_dict(complex_dict=v, separator=separator, final_key = current_key))
            
    return flatten_dict


def permute_channel_layout(
    image: torch.Tensor,
    target_format: Literal["**C", "C**"],
    n_channel: int | None = None,
) -> torch.Tensor:
    """
    Convert image layout between channel-last (**C) and channel-first (C**).

    "**C" means (H, W, C)
    "C**" means (C, H, W)
    Parameters
    ----------
    image: (torch.Tensor) the image to permute dimensions
    target_format (Literal['**c', 'c**']): define the final format of image, to adjust the permutation considering this
    n_channel (Optional[int]): for different type of images where nChannels is not standard as the minimum of args of [C, H, W]

    If the image is grayscale after squeeze, i.e. ndim <= 2,
    it is returned unchanged.
    """
    

    image = image.squeeze()

    if image.ndim <= 2:
        standard_log.info('the image is a grayscale, no need to permute it')
        return image

    if image.ndim != 3:
        raise ValueError(
            f"Expected a 2D or 3D image after squeeze, got shape {image.shape}"
        )

    d1, _, d3 = image.shape

    if n_channel is not None:
        if d1 == n_channel:
            current_format = "C**"

        elif d3 == n_channel:
            current_format = "**C"
        else:
            raise ValueError(
                f"Cannot find channel dimension with n_channel={n_channel} "
                f"in image shape {image.shape}"
            )
    else:
        # fallback euristico: il canale è la dimensione più piccola
        channel_axis = torch.argmin(torch.tensor(image.shape)).item()

        if channel_axis == 0:
            current_format = "C**"
        elif channel_axis == 2:
            current_format = "**C"
        else:
            raise ValueError(
                f"Channel axis appears to be in the middle for shape {image.shape}. "
                "Expected channel-first or channel-last."
            )

    if current_format == target_format:
        standard_log.info(f'the image shape {image.shape} is already in the format {target_format}')
        return image

    if current_format == "**C" and target_format == "C**":
        standard_log.info(f'the image shape {image.shape} will be transposed in the format {target_format}')
        return image.permute(2, 0, 1)

    if current_format == "C**" and target_format == "**C":
        standard_log.info(f'the image shape {image.shape} will be transposed in the format {target_format}')
        return image.permute(1, 2, 0)

    raise ValueError(f"Unsupported target_format: {target_format}")



def check_same_device(*objects, 
                      expected_device: torch.device | str | None = None,
                      ) -> torch.device:
    """
    Check that all tensors and nn.Module parameters inside the given objects
    are on the same device.

    Parameters
    ----------
    *objects
        Any objects containing torch.Tensor, nn.Module, dict, list, tuple, etc.
    expected_device : torch.device | str | None
        If given, all tensors must be on this device. And then move them if not
        If None, the first tensor/device found is used as reference.

    Returns
    -------
    torch.device
        The reference device.

    Raises
    ------
    RuntimeError
        If a device mismatch is found.
    """
    ref_device = torch.device(expected_device) if expected_device is not None else None
    problems: list[str] = []

    def visit(obj, path: str) -> None:
        nonlocal ref_device

        if obj is None:
            return

        if isinstance(obj, torch.Tensor):
            if ref_device is None:
                ref_device = obj.device
            elif obj.device != ref_device:
                problems.append(f"{path}: {obj.device} != {ref_device}")
            return

        if isinstance(obj, nn.Module):
            for name, param in obj.named_parameters(recurse=True):
                visit(param, f"{path}.{name}")
            for name, buffer in obj.named_buffers(recurse=True):
                visit(buffer, f"{path}.{name}")
            return

        if isinstance(obj, Mapping):
            for key, value in obj.items():
                visit(value, f"{path}[{repr(key)}]")
            return

        if isinstance(obj, (list, tuple)):
            for i, value in enumerate(obj):
                visit(value, f"{path}[{i}]")
            return

        # Ignora altri tipi: int, float, str, np.ndarray, ecc.

    for i, obj in enumerate(objects):
        visit(obj, f"arg{i}")

    if ref_device is None:
        raise RuntimeError("No torch.Tensor or nn.Module parameters found to check.")

    if problems:
        raise RuntimeError(
            "Device mismatch detected:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )

    return ref_device



# ======================================================================================
#                                     OBJS UTILITIES(converting their nature)
# ======================================================================================

def get_tensor(input: Any,
               )->torch.Tensor:
    '''
    Generate a Tensor from a general input, if the input type is not handled yet it ha to be write down

    Args:
        input (Any): General input whose tensor is in need

    Returns:
        torch.Tensor: corresponding tensor of input data
    '''
    
    if hasattr(input, 'image'):
        return torch.Tensor(input.image)
    if isinstance(input, np.ndarray):
        return torch.from_numpy(input.astype(np.float32))
    elif isinstance(input, torch.Tensor):
        return input
    else:
        standard_log.warning(f'the data type {type(input)} is yet to be handled, introduce a new block to us it correctly')
        raise TypeError(f'Unsupported Type{ type(input)}')



def tensor_img_rgb2bn(imgrgb: torch.Tensor | np.ndarray,
                      normalization_type: Literal['same',  'humansensitivity'] = 'same',
                      )->torch.Tensor:
    '''
    convert a RGB torch tensor image to a grayscale torch tensor image
    imgrgb in format (C, H, W) or  (B, C, H, W)
    normalization_type: 'same' to keep the same width for each channel
                        'humansensitivity' to use the human sensitivity ot each color channel: R: 0.2989, G: 0.5870, B: 0.1140
    '''
    if isinstance(imgrgb, np.ndarray):
        imgrgb = torch.from_numpy(imgrgb)
    img_rgb = imgrgb.squeeze().detach().cpu()
    if img_rgb.ndim == 2: # single channel image
        return img_rgb
    elif img_rgb.ndim == 3 and img_rgb.shape[0] == 3: # multi-channel image
        if normalization_type == 'same':
            img_bn = img_rgb.mean(dim=0)
        elif normalization_type == 'humansensitivity':
            rgb_to_gray_sensitivity = torch.tensor([0.2989, 0.5870, 0.1140])
            img_bn = torch.tensordot(permute_channel_layout(image=img_rgb, target_format='**C'), rgb_to_gray_sensitivity, dims=1)
        else:
            raise ValueError("normalization_type must be 'same' or 'humansensitivity'")
    else:
        raise ValueError(f'the input shape must be in (C, H, W) or (B, C, H, W), got {imgrgb.shape}')
    return img_bn