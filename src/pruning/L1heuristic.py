__all__ = ['L1Heuristic']

import tensorflow as tf

from .utils import *

"""
Heuristic of "Learning bothWeights and Connections for Efficient
Neural Networks" paper:
parameter: - model
           - kwargs: - prune_last(bool): decide if the score of the last layer is computed or not
                     - reset (bool): reset mask for the computation of the score
"""

def L1Heuristic(model,**kwargs):
    
    prune_last = kwargs["prune_last"]
    reset= kwargs["reset"]
    
    dico_score={}

    listlayer=getAllLayer(model)
    for layer in listlayer:
        if hasattr(layer,"kernel_mask"):
            if (prune_last or (prune_last==False and layer.last==False)):
                dico_score[layer]={}
                if reset:
                    dico_score[layer]["kernel"]=tf.math.abs(layer.kernel)
                    if layer.bias_mask is not None:
                        dico_score[layer]["bias"]=tf.math.abs(layer.bias)
                else:
                    dico_score[layer]["kernel"]=tf.math.abs(layer.kernel*layer.kernel_mask)
                    if layer.bias_mask is not None:
                        dico_score[layer]["bias"]=tf.math.abs(layer.bias*layer.bias_mask)
                        
    return dico_score
