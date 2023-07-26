# -*- coding:UTF-8 -*-

import logging

"""
logger
    
"""


def creat_logger(log_dir: str, name: str = "regformer"):
    """

    :param log_dir
    :param name:
    :return:
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + '/train_log.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def log_print(logger, information: str = 'logger and print'):
    """

    :param logger
    :param information
    """
    print(information)
    logger.info(information)
