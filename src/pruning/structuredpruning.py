__all__ = ['StructuredPruning']

import tensorflow as tf
import numpy as np

from .utils import *

"""
Structured pruning function:
parameter: - model
           - heuristic_function: heuristic (L1 norm, SNIP, Synflow, FPGM)
           - sparsity (float): percentage of weights whose values will be 0 after the function
           - previous sparsity (float): percentage of weights whose values are 0 before the function
           - layer_wise (bool): local or global pruning
           - aggregate (string): aggregation of score across a filter (or neuron) : possible value are "mean", "sum" or None 
           - iterative (integer) : number of iteration used in the cycle ranking/pruning)
           - kwargs: used for the heuristic computation
"""
    
def StructuredPruning(model,heuristic_function,sparsity=0.2,previous_sparsity=0.0,layer_wise=True,aggregate="mean",iterative=1,**kwargs):

    if sparsity==previous_sparsity:
        return model
      
    assert iterative>0,"iterative value should be superior to 0"   
    for i in range(iterative):
    
        power = (iterative-(i+1))/iterative
        new_sparsity = sparsity+1 - (sparsity-previous_sparsity)**power
        print(new_sparsity)
        dico_score = heuristic_function(model,**kwargs)
        
        for layer,score in dico_score.items():
            score.pop('bias', None)
            for key in score.keys():
                if aggregate == "mean":
                    dico_score[layer][key]=tf.math.reduce_sum(dico_score[layer][key], axis=list(range(tf.rank(dico_score[layer][key])-1)))/tf.cast(tf.math.reduce_prod(dico_score[layer][key].shape[:-1]),dtype=tf.float32)    
                elif aggregate == "sum":
                    dico_score[layer][key]=tf.math.reduce_sum(dico_score[layer][key], axis=list(range(tf.rank(dico_score[layer][key])-1)))
                 
        if layer_wise:
            for layer,score in dico_score.items():
                threshold = np.percentile(score["kernel"].numpy(),new_sparsity)
                score_mask = tf.cast(tf.where(score["kernel"]<=threshold,0,1),dtype=tf.float32)
                layer.kernel_mask.assign(tf.ones_like(layer.kernel_mask)*score_mask)
                if layer.bias_mask is not None:
                    layer.bias_mask.assign(tf.ones_like(layer.bias_mask)*score_mask)
        else:
            score_vector = tf.concat([score["kernel"] for score in dico_score.values() if "kernel" in score],-1)
            threshold = np.percentile(score_vector.numpy(),new_sparsity)
            for layer,score in dico_score.items():
                score_mask = tf.cast(tf.where(score["kernel"]<=threshold,0,1),dtype=tf.float32)
                layer.kernel_mask.assign(tf.ones_like(layer.kernel_mask)*score_mask)
                if layer.bias_mask is not None:
                    layer.bias_mask.assign(tf.ones_like(layer.bias_mask)*score_mask)
                
    return model
