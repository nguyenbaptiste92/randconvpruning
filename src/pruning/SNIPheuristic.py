__all__ = ['SNIPHeuristic']

import tensorflow as tf
import numpy as np

from .utils import *

###############################################################################################################################################
#STEP TO COMPUTE THE GRADIENT FOR SNIP HEURISTIC (MONOGPU AND MULTIGPU)
###############################################################################################################################################

def compute_gradients_step(model, elem, prediction_loss, constitency_loss, mode="simple", ratio=10, rand_module=None, rand_iter=0, multi_gpu=False):

    inputs = elem["image"]
    target = elem["label"]
    with tf.GradientTape() as tape:
        pred_loss = 0.0
        inv_loss = 0.0
        if mode=="simple":
            prediction = model(inputs,training=False)["label"]
            if multi_gpu:
                pred_loss += tf.reduce_sum(prediction_loss(target,prediction))
            else:
                pred_loss += prediction_loss(target,prediction)
        
        else:
            assert rand_module is not None and rand_iter>0
            list_prediction = []
            for i in range(rand_iter):
                input_batch = rand_module(inputs)
                prediction = model(input_batch,training=False)["label"]
                list_prediction.append(prediction)
            
            if mode == "rand_conv" or mode == "rand_conv_total":
                for prediction in list_prediction:
                    if multi_gpu:
                        pred_loss += tf.reduce_sum(prediction_loss(target,prediction))
                    else:
                        pred_loss += prediction_loss(target,prediction)
                        
            if mode == "rand_conv_consistency" or mode == "rand_conv_total":
                p_mixture = tf.math.add_n(list_prediction)/rand_iter
                for prediction in list_prediction:
                    if multi_gpu:
                        inv_loss+= tf.reduce_sum(constitency_loss(target,prediction))
                    else:
                        inv_loss += constitency_loss(target,prediction)
        total_loss = pred_loss + ratio * inv_loss
        
    return tape.gradient(total_loss, model.trainable_variables,unconnected_gradients=tf.UnconnectedGradients.NONE)
    
@tf.function    
def distributed_compute_gradients_step(strategy, model, elem, prediction_loss, constitency_loss, mode="simple", ratio=10, rand_module=None, rand_iter=0):
    
    per_replica_grad = strategy.run(compute_gradients_step, args=(model, elem, prediction_loss, constitency_loss, mode, ratio, rand_module, rand_iter))

    return [strategy.reduce("SUM", element,axis=None) for element in per_replica_grad]


###############################################################################################################################################
#SNIP HEURISTIC
###############################################################################################################################################
"""
Heuristic of "SNIP: Single-shot Network Pruning based on Connection Sensitivity" paper:
    Input:
        model : model to prune (class tf.keras.Model)
        kwargs :  -prune_last (bool): decide if the score of the last layer is computed or not
                  -dataset (tf.data.Dataset with input and output): dataset on which the network will be training
                  -mini_batches (integer): number of batches using for the calcul of saliency (0 will use the whole dataset)
                  -multi_gpu (bool): use multiple gpus to speed up the computations
                  -strategy (tf.distribute.MirroredStrategy) : strategy use for multi-gpu
                  -mode (string): mode to compute gradients: "simple" (compute gradients without random convolutions),"rand_conv" (compute gradients with random convolutions with the prediction loss), "rand_conv_consistency" (compute gradients with random convolutions with the consistency loss), "rand_conv_total" (compute gradients with random convolutions with the prediction loss and the consistency loss)
                  -rand_conv (RandConvModule) : random_convolutions module used.
                  -prediction_loss (tf.keras.losses): prediction_loss (in general, tf.keras.losses.CategoricalCrossentropy)
                  -consistency_loss (tf.keras.losses): consistency loss (in general, tf.keras.losses.KLDivergence)
                  -ratio (float): ratio between prediction loss and consistency loss 
                  -rand_iter (integer) : number of random convolutions use for data_augmentation
"""


def SNIPHeuristic(model,**kwargs):

    prune_last = kwargs["prune_last"]
    dataset = kwargs["dataset"]
    mini_batches= kwargs["mini_batches"]
    multi_gpu = kwargs["multi_gpu"]
    strategy = kwargs["strategy"]
    mode = kwargs["mode"]
    rand_conv = kwargs["rand_conv"]
    prediction_loss = kwargs["prediction_loss"]
    constitency_loss = kwargs["constitency_loss"]
    ratio = kwargs["ratio"]
    rand_iter= kwargs["rand_iter"]

      
    dico_variable=getLinkLayerTrainableVariable(model)
    #Compute SNIP score of all trainable variables
    cumulated_grads = [tf.zeros_like(weight, dtype=tf.dtypes.float32, name=weight.name) for weight in model.trainable_variables]
    for i, elem in enumerate(dataset):
        if multi_gpu:
            grads = distributed_compute_gradients_step(strategy, model, elem, prediction_loss, constitency_loss, mode, ratio, rand_conv, rand_iter)
        else:
            grads = compute_gradients_step(model, elem, prediction_loss, constitency_loss, mode, ratio, rand_conv, rand_iter)
        cumulated_grads = [g+tf.math.abs(ag) for g, ag in zip(cumulated_grads,grads)]
        if i>=mini_batches: break
        
    #Filter the trainable variables
    dico_score={}
    for variable,grad in zip(model.trainable_variables,cumulated_grads):
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
