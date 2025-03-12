__all__ = ['prune']

import numpy as np

from .L1heuristic import L1Heuristic
from .SNIPheuristic import SNIPHeuristic
from .Synflowheuristic import SynflowHeuristic
from .FPGMheuristic import FPGMHeuristic

from .unstructuredpruning import UnstructuredPruning
from .structuredpruning import StructuredPruning

algorithm_map = ['L1','L1Filter','SNIP','SNIPFilter','Synflow','SynflowFilter','FPGMFilter']

"""
helper function in randconv_trainer.py to prune a network
arguments: -model: model to prune
           - sparsity (float): percentage of weights whose values will be 0 after the function
           - previous sparsity (float): percentage of weights whose values are 0 before the function
           - dataset (tf.data.Dataset with input and output): dataset used for SNIP computations
           - trainer (trainer of randconv_trainer.py): usefull for its variable args which contain usefull variables
"""
    
def prune(model,sparsity,previous_sparsity,dataset,trainer):

    assert trainer.args.pruning_algorithm in algorithm_map
    if trainer.args.pruning_algorithm=='L1':
        kwargs={"prune_last":trainer.args.prune_last,"reset":trainer.args.p_reset}
        new_model = UnstructuredPruning(model,L1Heuristic,sparsity,layer_wise=trainer.args.p_layerwise,**kwargs)
        
    elif trainer.args.pruning_algorithm=='L1Filter':
        kwargs={"prune_last":trainer.args.prune_last,"reset":trainer.args.p_reset}
        new_model = StructuredPruning(model,L1Heuristic,sparsity,layer_wise=trainer.args.p_layerwise,**kwargs)
        
    elif trainer.args.pruning_algorithm=='SNIP':
        kwargs={"prune_last":trainer.args.prune_last,"dataset":dataset,"mini_batches":trainer.args.snip_batch if trainer.args.snip_batch>0 else np.inf,"multi_gpu":trainer.args.multi_gpu,"strategy":trainer.strategy if trainer.args.multi_gpu else None, "mode":trainer.args.snip_mode, "rand_conv":trainer.rand_module,"prediction_loss" : trainer.criterion,"constitency_loss" : trainer.invariant_criterion, "ratio" :trainer.args.consistency_loss_w,"rand_iter":trainer.args.snip_iter}
        new_model = UnstructuredPruning(model,SNIPHeuristic,sparsity,layer_wise=trainer.args.p_layerwise,previous_sparsity=previous_sparsity,iterative=trainer.args.prune_iterative,**kwargs)
        
    elif trainer.args.pruning_algorithm=='SNIPFilter':
        kwargs={"prune_last":trainer.args.prune_last,"dataset":dataset,"mini_batches":trainer.args.snip_batch if trainer.args.snip_batch>0 else np.inf,"multi_gpu":trainer.args.multi_gpu,"strategy":trainer.strategy if trainer.args.multi_gpu else None, "mode":trainer.args.snip_mode, "rand_conv":trainer.rand_module,"prediction_loss" : trainer.criterion,"constitency_loss" : trainer.invariant_criterion, "ratio" :trainer.args.consistency_loss_w,"rand_iter":trainer.args.snip_iter}
        new_model = StructuredPruning(model,SNIPHeuristic,sparsity,layer_wise=trainer.args.p_layerwise,previous_sparsity=previous_sparsity,iterative=trainer.args.prune_iterative,**kwargs)
    
    elif trainer.args.pruning_algorithm=='Synflow':
        kwargs={"prune_last":trainer.args.prune_last,"multi_gpu":trainer.args.multi_gpu,"strategy":trainer.strategy if trainer.args.multi_gpu else None}
        new_model = UnstructuredPruning(model,SynflowHeuristic,sparsity,layer_wise=trainer.args.p_layerwise,previous_sparsity=previous_sparsity,iterative=trainer.args.prune_iterative,**kwargs)
        
    elif trainer.args.pruning_algorithm=='SynflowFilter':
        kwargs={"prune_last":trainer.args.prune_last,"multi_gpu":trainer.args.multi_gpu,"strategy":trainer.strategy if trainer.args.multi_gpu else None}
        new_model = StructuredPruning(model,SynflowHeuristic,sparsity,layer_wise=trainer.args.p_layerwise,previous_sparsity=previous_sparsity,iterative=trainer.args.prune_iterative,**kwargs)
    elif trainer.args.pruning_algorithm=='FPGMFilter':
        kwargs={"prune_last":trainer.args.prune_last}
        new_model = StructuredPruning(model,FPGMHeuristic,sparsity,layer_wise=True,aggregate="no_aggregate",**kwargs)
    else:
        new_model = model
    return new_model
