__all__ = ['FPGMHeuristic']

import tensorflow as tf
import numpy as np
from scipy.spatial import distance

from .utils import *

"""
Heuristic of "Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration" paper:
parameter: - model
           - kwargs: in kwargs, the usefull parameter is "prune_last" which decide if the score of the last layer is computed or not
"""

def FPGMHeuristic(model,**kwargs):
    
    prune_last = kwargs["prune_last"]
    
    dico_score={}

    listlayer=getAllLayer(model)
    i=0
    for layer in listlayer:
        if hasattr(layer,"kernel_mask"):
            if (prune_last or (prune_last==False and layer.last==False)):
                dico_score[layer]={}
                #Filter already pruned filters
                shape=layer.kernel_mask.shape[-1]
                pruned_filter_index = tf.math.reduce_sum(tf.math.abs(layer.kernel*layer.kernel_mask),axis=list(range(tf.rank(layer.kernel)-1)))
                indices = tf.where(tf.math.not_equal(pruned_filter_index, 0.0, name=None))
                filter_kernel = tf.squeeze(tf.gather(layer.kernel, indices, axis=-1),axis=-1)
                flatten_filter_kernel = tf.reshape(filter_kernel,(filter_kernel.shape[-1],-1))
                similar_matrix = distance.cdist(flatten_filter_kernel, flatten_filter_kernel, 'euclidean')
                score = tf.cast(tf.math.reduce_sum(tf.math.abs(similar_matrix),axis = 0),tf.float32)
                dico_score[layer]["kernel"]=-tf.tensor_scatter_nd_update(np.inf*tf.ones([shape],dtype=tf.float32),indices,score)
                        
    return dico_score