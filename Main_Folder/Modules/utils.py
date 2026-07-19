# FIle containing function f trasversal utility
#import sys
#import os
from pathlib import Path
#import numpy as np
#import yaml
#from typing import Tuple, Literal
#import logging



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
            return path

    raise FileNotFoundError(
        f"Project root '{project_name}' not found." 
    )



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