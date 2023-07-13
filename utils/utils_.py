from os.path import join as pjoin
from os.path import dirname as getdir
from os.path import basename as getbase
from os.path import splitext
from tqdm.auto import tqdm
import math, random
import numpy as np
import time
from glob import glob
import csv
import torch
from typing import List
import logging
import os

def write_dict_to_csv(data: dict, file_path):
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)

def become_deterministic(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dict_add(dictionary:dict, key, value, acc='list'):
    if key not in dictionary.keys():
        if acc=='list':
            dictionary[key] = []
        elif acc=='set':
            dictionary[key] = set()
        else:
            assert False, 'only list or set'
    dictionary[key] += [value]

def get_str_after_substring(text:str, substring:str):
    index = text.find(substring)
    if index >= 0:
        next_char = text[index + len(substring):]
        return substring + next_char
    else:
        return None

def fn_comb(kwargs: List):
    def comb(X):
        for fn in kwargs:
            X = fn(X)
        return X
    return comb

class measure_time():
    def __init__(self):
        pass
    def __enter__(self):
        self.start_time = time.time()
    def __exit__(self, type, value, traceback):
        print('time elapsed', time.strftime("%H:%M:%S", time.gmtime(time.time() - self.start_time)))


class printc:
    """colorful print, but now I want colorul logging to show the message"""
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    END = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    BLACK = "\033[1;30m"
    RED = "\033[1;31m"
    GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[1;34m"
    PURPLE = "\033[1;35m"
    CYAN = "\033[1;36m"
    WHITE = "\033[1;37m"

    @staticmethod
    def blue(*text):
        printc.uni(printc.BLUE, text)
    @staticmethod
    def green(*text):
        printc.uni(printc.GREEN, text)
    @staticmethod
    def yellow(*text):
        printc.uni(printc.YELLOW, text)
    @staticmethod
    def red(*text):
        printc.uni(printc.RED, text)
    @staticmethod
    def uni(color, text:tuple):
        print(color + ' '.join([str(x) for x in text]) + printc.END)

class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


