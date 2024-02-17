# -*- coding:UTF-8 -*-

import logging

# author:Zhiheng Feng
# contact: fzhsjtu@foxmail.com
# datetime:2021/10/21 20:20
# software: PyCharm

"""
文件说明： logger
    
"""


def creat_logger(log_dir: str, name: str = "pwclonet"):
    """

    :param log_dir: 存放日志文件的路径
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

    :param logger: 日志文件
    :param information: 日志信息
    :return: 日志记录模块，用于日志记录
    """
    print(information)
    logger.info(information)
