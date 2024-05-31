__all__ = ['get_dataset','dataset_std','dataset_mean','dataset_size']

import tensorflow as tf

from .transforms import Cast,Reshape,Normalize,GreyToColor
from .standard_dataset import get_tfds_dataset
from .mnist_10k import get_MNIST10K
from .usps import get_USPS
from .hhar import get_HHAR
from .pamap2 import get_PAMAP2
from .opportunity import get_opportunity
from .realworld import get_realworld

digits = 'digits'
mnist = 'mnist'
mnist10k = 'mnist10k'
mnist_c = 'mnist_c'
svhn = 'svhn'
usps = 'usps'

har = 'human activity recognition'
oppor = 'opportunity'
hhar = 'hhar'
real = 'realworld'
pamap ='pamap2'

dataset_std = {digits: (0.5, 0.5, 0.5)}
dataset_mean = {digits: (0.5, 0.5, 0.5)}
dataset_size = {digits: (32,32,3)}

"""
helper function in train_digits.py and train_har.py to load the dataset
arguments: -name (string): name of the network
           -args (list): arguments usefull 
"""

def get_dataset(name, **kwargs):
    assert 'path' in kwargs, "Need to declare a path to use this function."
    print(name)
    if name == mnist:
        if 'transform' not in kwargs:
            transform=[Cast(10),Rescale(255),Reshape(dataset_size[digits][:-1]),GreyToColor(),Normalize(dataset_mean[digits], dataset_std[digits])]
            kwargs['transform'] = transform
        data = get_tfds_dataset(name,**kwargs)

    elif name == mnist10k:
        if 'transform' not in kwargs:
            transform=[Cast(10),Rescale(255),Reshape(dataset_size[digits][:-1]),GreyToColor(),Normalize(dataset_mean[digits], dataset_std[digits])]
            kwargs['transform'] = transform
        data = get_MNIST10K(**kwargs)

    elif name == svhn:
        if 'transform' not in kwargs:
            transform=[Cast(10),Rescale(255),Normalize(dataset_mean[digits], dataset_std[digits])]
            kwargs['transform'] = transform
        name="svhn_cropped"
        data = get_tfds_dataset(name,**kwargs)

    elif name == usps:
        if 'transform' not in kwargs:
            transform=[Cast(10),Rescale(255),Reshape(dataset_size[digits][:-1]),GreyToColor(),Normalize(dataset_mean[digits], dataset_std[digits])]
            kwargs['transform'] = transform
        data = get_USPS(**kwargs)

    elif name == mnist_c:
        if 'transform' not in kwargs:
            transform=[Cast(10),Rescale(255),Reshape(dataset_size[digits][:-1]),GreyToColor(),Normalize(dataset_mean[digits], dataset_std[digits])]
            kwargs['transform'] = transform
        name="mnist_corrupted"
        data = get_tfds_dataset(name,**kwargs)
        
        
        
    elif name == oppor:
        if 'transform' not in kwargs:
            transform=[Cast(14)]
            kwargs['transform'] = transform
        data = get_opportunity(**kwargs)
    
    elif name == hhar:
        if 'transform' not in kwargs:
            transform=[Cast(14)]
            kwargs['transform'] = transform
        data = get_HHAR(**kwargs)
        
    elif name == real:
        if 'transform' not in kwargs:
            transform=[Cast(14)]
            kwargs['transform'] = transform
        data = get_realworld(**kwargs)
        
    elif name == pamap:
        if 'transform' not in kwargs:
            transform=[Cast(14)]
            kwargs['transform'] = transform
        data = get_PAMAP2(**kwargs)
        
        
        
    else:
        raise NotImplementedError('{} data does not exists'.format(name))
    return data