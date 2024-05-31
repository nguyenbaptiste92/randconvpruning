__all__ = ['compute_synflow_grad','distributed_compute_synflow_grad','SynflowHeuristic']

import tensorflow as tf
import numpy as np

from .utils import *

###############################################################################################################################################
#STEP TO COMPUTE THE GRADIENT FOR SYNFLOW HEURISTIC (MONOGPU AND MULTIGPU)
###############################################################################################################################################

def compute_synflow_grad(model):

    # Abs weight in model
    sign = [tf.math.sign(variable) for variable in model.trainable_variables]
    for variable in model.trainable_variables:
        variable.assign(tf.math.abs(variable))
    # Create input
    inputs_shape = [elem.shape for elem in model.inputs]
    inputs = [tf.ones([10]+elem[1:]) for elem in inputs_shape]
    
    # Compute gradient
    with tf.GradientTape() as tape:
        prediction = model(inputs,training=False)["output"] 
    grad = tape.gradient(prediction, model.trainable_variables,unconnected_gradients=tf.UnconnectedGradients.NONE)
    #reset model
    for variable,s in zip(model.trainable_variables,sign):
        variable.assign(s*variable)
    return grad
        
@tf.function    
def distributed_compute_synflow_grad(strategy,model):
    per_replica_grad = strategy.run(compute_synflow_grad, args=[model])
    grad = [strategy.reduce("SUM", element,axis=None) for element in per_replica_grad]
    return grad
    
    
###############################################################################################################################################
#SYNFLOW HEURISTIC
###############################################################################################################################################
"""
Heuristic of "SNIP: Single-shot Network Pruning based on Connection Sensitivity" paper:
    Input:
        model : model to prune (class tf.keras.Model)
        kwargs : -prune_last (bool): decide if the score of the last layer is computed or not
                  -multi_gpu (bool): use multiple gpus to speed up the computations
                  -strategy (tf.distribute.MirroredStrategy) : strategy used for multi-gpu
"""
    
def SynflowHeuristic(model,**kwargs):

       
    prune_last = kwargs["prune_last"]
    multi_gpu = kwargs["multi_gpu"]
    strategy = kwargs["strategy"]
       
    dico_variable=getLinkLayerTrainableVariable(model)
        
    if multi_gpu:
        grads = distributed_compute_synflow_grad(strategy,model)
    else:
        grads = compute_synflow_grad(model)
    #print(grads)
        
    #Filter the trainable variables
    dico_score={}
    for variable,grad in zip(model.trainable_variables,grads):
        layer = dico_variable[variable.name]
        if hasattr(layer,"kernel_mask"):
            if (prune_last or (prune_last==False and layer.last==False)):
                if layer not in dico_score:
                    dico_score[layer]={}
                if variable.shape.is_compatible_with(layer.kernel.shape):
                    dico_score[layer]["kernel"] = tf.math.abs(grad*variable)
                else:
                    dico_score[layer]["bias"] = tf.math.abs(grad*variable)
    
    return dico_score