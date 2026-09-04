"""Copyrigth (c) 2024 by R. Eliasy

Author: Renato Eliasy
Contact: renatoeliasy@gmail.com
Date: 04-09-2026

Introduction of the __init__.py file to recognize the directory as a module"""
# TEST IMPORT FOR AVOID CIRCULAR IMPORTS

from Main_Folder.Modules  import common, download, utils

from Main_Folder.Modules.configuration_setting  import logger_configuration, yaml_configuration

from Main_Folder.Modules.framework  import (
    data_classes,
    evaluations,
    img_io,
    metrics,
    postprocessing,
    preprocessing,
    visualization
)

from Main_Folder.Modules.framework.registration import (
    algorithms,
    loops,
    methods,
    networks
)

print('All modules imported successfully.')