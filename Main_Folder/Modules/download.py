from pathlib import Path
import requests
import zipfile
import py7zip as py7z
import os
import sys
from Main_Folder.Modules.configuration_setting.logger_configuration import get_logger
from Main_Folder.configparser import Config
from Main_Folder.Modules.utils  import get_root_path

standard_log = get_logger(__name__)
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
    try:
        with open(file_path, 'r') as f:
            standard_log.warning(f'the {file_path} already exists')
    except FileNotFoundError:
        request = requests.get(url=url)
        if not request.ok:
            standard_log.error(f'{url} got a problem due to {request.status_code} code')
        else:
            with open(file_path, 'wb') as f:
                f.write(request.content)
    return

def import_trial():
    print('successful')
    return None


if __name__ == "__main__":
    print(config_file_path)
