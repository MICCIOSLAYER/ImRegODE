# FIle containing function f trasversal utility
import sys
import os
from pathlib import Path
#import numpy as np
#import yaml
#from typing import Tuple, Literal
from Main_Folder.Modules.configuration_setting.logger_configuration import get_logger
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


# ======================================================================================
#                                     OBJS UTILITIES
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