import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import argparse
import functools
import inspect

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import datasetops as do
from torchvision.datasets import USPS

def print_imported_modules():
    import sys
    for name, val in sorted(sys.modules.items()):
        if(hasattr(val, '__version__')): 
            print(val.__name__, val.__version__)
#        else:
#            print(val.__name__, "(unknown version)")
def print_sys_info():
    import sys
    import platform
    print(sys.version)
    print(platform.uname())


f = open('environment.txt','w')
sys.stdout = f

print_sys_info()
print_imported_modules()