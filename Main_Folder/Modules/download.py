from pathlib import Path
import requests
import zipfile
import py7zip as py7z
import os
import sys

from Main_Folder.configparser import Config

config_file_path = Path(__file__).resolve().parents[2] / "config.txt"
config = Config(config_file_path)
links = config.get_link()
fire_dataset = links["FIRE_DATASET"]


def download_file(
    url: str,
    file_path: Path,
) -> None:
    """
    Download a file from a URL and save it to a specific path.

    Parameters
    ----------
    url : str
        The URL to download the file.
    file_path : Path
        The path to save the file.

    Returns
    -------
    None

    """

    pass

def import_trial():
    print('successful')
    return None
if __name__ == "__main__":
    print(config_file_path)
