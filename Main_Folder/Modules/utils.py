# FIle containing function f trasversal utility
import sys
import os
from pathlib import Path



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
        f"Project root '{project_name}' not found." # FIXME search in daughters dirs
    )
