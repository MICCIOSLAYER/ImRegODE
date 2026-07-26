# script to manage and to categorize all the input/output functions
from Main_Folder.Modules.configuration_setting.logger_configuration import get_logger


import os
import py7zr
import requests
import zipfile
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

# ============= CATEGORIZE ELEMENTS FNS HERE ==================