import logging
from pathlib import Path
from typing import Optional, Union, Literal

LOGGING_LEVEL = Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

def set_logger(
    level:  LOGGING_LEVEL = 'DEBUG',
    log_file: Optional[Union[str, Path]] = None
) -> logging.Logger:
    '''
    a function to set the logger and logging for each file/script, it has to be called at the start of each script if used:
    the LEVEL OF PRIORITY: DEBUG < INFO < WARNING < ERROR < CRITICAL. As stated logging.DEBUG prints also logging.INFO, 
    but logging.WARNING doesn't print logging.DEBUG

    Args:
        level (int, optional): Define the level of verbosity. Defaults to logging.INFO.
        log_file (str | None, optional): The log file where most/all of the log will be saved. Defaults to None.

    Returns:
        logging.Logger: to be used as log_for_this_file = set_logger(..) and then log_for_this_file.DEBUG('message')
    '''
    
    # reset (importante nei notebook)
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    logger = logging.getLogger()
    logger.handlers.clear()

    log_level = getattr(logging, level.upper())
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        '[%(levelname)s] %(name)s - %(message)s'
    )

    # console
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # file opzionale
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def get_logger(name:str):
    return logging.getLogger(name)