"""Copyrigth (c) 2024 by R. Eliasy

Author: Renato Eliasy
Contact: renatoeliasy@gmail.com
Date: 06-12-2024

Introduction of the __init__.py file to recognize the directory as a module
Configuration of the config parser object"""

import configparser
from pathlib import Path


class Config:
    """A class of objects that can read data from a configuration file."""

    def __init__(
        self,
        file_path: Path,
    ) -> None:
        """
        Contructor of the class.

        Parameters
        ----------
        file_path : Path
            The path to the configuration file.

        Returns
        -------
        None

        """
        self.config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())
        self.config.read(file_path)

    def get_link(
        self,
    ) -> dict[str, str]:
        """
        Return a dictionary with the cutoffs for thresholding the attention
        matrices and for binarizing the contact map.

        Returns
        -------
        dict[str, str]
            The identifier and the link to download the corresponding dataset.

        """
        return {
            "FIRE_DATASET": str(
                self.config.get("dataset", "FIRE")
            ),
        }

    def get_paths(
        self,
    ) -> dict[str, str]:
        """
        Return a dictionary with the paths to folders to store the files.

        Returns
        -------
        dict[str, str]
            The identifier and the paths to the corresponding folder.

        """
        return {
            "IMAGES_FOLDER": self.config.get("paths", "IMAGES"),
            "TEXT_FOLDER": self.config.get("paths", "TEXT"),
            "DATASET_FOLDER": self.config.get("paths", "DATASET"),
        }
