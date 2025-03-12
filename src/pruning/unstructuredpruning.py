__all__ = ['UnstructuredPruning']

import tensorflow as tf
import numpy as np

"""
Unstructured pruning function:
parameter: - model
           - heuristic_function: heuristic (L1 norm, SNIP, Synflow)
           - sparsity (float): percentage of weights whose values will be 0 after the function
           - previous sparsity (float): percentage of weights whose values are 0 before the function
           - layer_wise (bool): local or global pruning
           - iterative (integer) : number of iteration used in the cycle ranking/pruning)
           - kwargs: used for the heuristic computation
"""

def UnstructuredPruning(model,heuristic_function,sparsity=20.0,previous_sparsity=0.0,layer_wise=True,iterative=1,**kwargs):
    
    if sparsity==previous_sparsity:
        return model
    
    assert iterative>0,"iterative value should be superior to 0"  
    for i in range(iterative):
        power = (iterative-(i+1))/iterative
        new_sparsity = sparsity+1 - (sparsity-previous_sparsity)**power
        print(new_sparsity)
        
        #Compute score with heuristic
        dico_score = heuristic_function(model,**kwargs)
          
        #Prune        
        if layer_wise:
            for layer,dico in dico_score.items():
                list_score = list(dico.values())
                list_score = [tf.reshape(x, [-1]) for x in list_score]
                score_vector = tf.concat(list_score,-1)
                threshold = np.percentile(score_vector.numpy(),new_sparsity)
                new_kernel_mask = tf.cast(tf.where(dico["kernel"]<=threshold,0,1),dtype=tf.float32)
                layer.kernel_mask.assign(new_kernel_mask)
                if layer.bias_mask is not None:
                    new_bias_mask = tf.cast(tf.where(dico["bias"]<=threshold,0,1),dtype=tf.float32)
                    layer.bias_mask.assign(new_bias_mask)
        else:
            list_score = [dico["kernel"] for dico in dico_score.values()]+[dico.get("bias") for dico in dico_score.values()]
            list_score = [x for x in list_score if x is not None]
            list_score = [tf.reshape(x, [-1]) for x in list_score]
            score_vector = tf.concat(list_score,-1)
            threshold = np.percentile(score_vector.numpy(),new_sparsity)
            for layer,dico in dico_score.items():
                new_kernel_mask = tf.cast(tf.where(dico["kernel"]<=threshold,0,1),dtype=tf.float32)
                layer.kernel_mask.assign(new_kernel_mask)
                if layer.bias_mask is not None:
                    new_bias_mask = tf.cast(tf.where(dico["bias"]<=threshold,0,1),dtype=tf.float32)
                    layer.bias_mask.assign(new_bias_mask)
        
                
    return model