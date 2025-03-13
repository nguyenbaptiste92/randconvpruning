__all__ = ["prune"]

from typing import Any

import numpy as np
import tensorflow as tf

from .fpgm_heuristic import fpgm_heuristic
from .l1_heuristic import l1_heuristic
from .snip_heuristic import snip_heuristic
from .structuredpruning import structuredpruning
from .synflow_heuristic import synflow_heuristic
from .unstructuredpruning import unstructuredpruning

algorithm_map = [
    "L1",
    "L1Filter",
    "SNIP",
    "SNIPFilter",
    "Synflow",
    "SynflowFilter",
    "FPGMFilter",
]


def prune(
    model: tf.keras.Model,
    sparsity: float,
    previous_sparsity: float,
    dataset: tf.data.Dataset,
    trainer: Any,
) -> tf.keras.Model:
    """
    Summary: prune model

    :param model: model to prune
    :type model: tf.keras.Model
    :param sparsity: percentage of weights whose values will be 0
    after the function
    :type sparsity: float
    :param previous_sparsity: percentage of weights whose values are 0
    before the function
    :type previous_sparsity: float
    :param dataset: dataset used for SNIP computations
    :type dataset: tf.data.Dataset
    :param trainer: struct which contain usefull variables
    :type trainer: Any
    :return: pruned model
    :rtype: tf.keras.Model
    """
    if trainer.args.pruning_algorithm == "L1":
        kwargs = {
            "prune_last": trainer.args.prune_last,
            "reset": trainer.args.p_reset,
        }
        new_model = unstructuredpruning(
            model,
            l1_heuristic,
            sparsity,
            layer_wise=trainer.args.p_layerwise,
            **kwargs,
        )

    elif trainer.args.pruning_algorithm == "L1Filter":
        kwargs = {
            "prune_last": trainer.args.prune_last,
            "reset": trainer.args.p_reset,
        }
        new_model = structuredpruning(
            model,
            l1_heuristic,
            sparsity,
            layer_wise=trainer.args.p_layerwise,
            **kwargs,
        )

    elif trainer.args.pruning_algorithm == "SNIP":
        kwargs = {
            "prune_last": trainer.args.prune_last,
            "dataset": dataset,
            "mini_batches": trainer.args.snip_batch
            if trainer.args.snip_batch > 0
            else np.inf,
            "multi_gpu": trainer.args.multi_gpu,
            "strategy": trainer.strategy if trainer.args.multi_gpu else None,
            "mode": trainer.args.snip_mode,
            "rand_conv": trainer.rand_module,
            "prediction_loss": trainer.criterion,
            "constitency_loss": trainer.invariant_criterion,
            "ratio": trainer.args.consistency_loss_w,
            "rand_iter": trainer.args.snip_iter,
        }
        new_model = unstructuredpruning(
            model,
            snip_heuristic,
            sparsity,
            layer_wise=trainer.args.p_layerwise,
            previous_sparsity=previous_sparsity,
            iterative=trainer.args.prune_iterative,
            **kwargs,
        )

    elif trainer.args.pruning_algorithm == "SNIPFilter":
        kwargs = {
            "prune_last": trainer.args.prune_last,
            "dataset": dataset,
            "mini_batches": trainer.args.snip_batch
            if trainer.args.snip_batch > 0
            else np.inf,
            "multi_gpu": trainer.args.multi_gpu,
            "strategy": trainer.strategy if trainer.args.multi_gpu else None,
            "mode": trainer.args.snip_mode,
            "rand_conv": trainer.rand_module,
            "prediction_loss": trainer.criterion,
            "constitency_loss": trainer.invariant_criterion,
            "ratio": trainer.args.consistency_loss_w,
            "rand_iter": trainer.args.snip_iter,
        }
        new_model = structuredpruning(
            model,
            snip_heuristic,
            sparsity,
            layer_wise=trainer.args.p_layerwise,
            previous_sparsity=previous_sparsity,
            iterative=trainer.args.prune_iterative,
            **kwargs,
        )

    elif trainer.args.pruning_algorithm == "Synflow":
        kwargs = {
            "prune_last": trainer.args.prune_last,
            "multi_gpu": trainer.args.multi_gpu,
            "strategy": trainer.strategy if trainer.args.multi_gpu else None,
        }
        new_model = unstructuredpruning(
            model,
            synflow_heuristic,
            sparsity,
            layer_wise=trainer.args.p_layerwise,
            previous_sparsity=previous_sparsity,
            iterative=trainer.args.prune_iterative,
            **kwargs,
        )

    elif trainer.args.pruning_algorithm == "SynflowFilter":
        kwargs = {
            "prune_last": trainer.args.prune_last,
            "multi_gpu": trainer.args.multi_gpu,
            "strategy": trainer.strategy if trainer.args.multi_gpu else None,
        }
        new_model = structuredpruning(
            model,
            synflow_heuristic,
            sparsity,
            layer_wise=trainer.args.p_layerwise,
            previous_sparsity=previous_sparsity,
            iterative=trainer.args.prune_iterative,
            **kwargs,
        )
    elif trainer.args.pruning_algorithm == "FPGMFilter":
        kwargs = {"prune_last": trainer.args.prune_last}
        new_model = structuredpruning(
            model,
            fpgm_heuristic,
            sparsity,
            layer_wise=True,
            aggregate="no_aggregate",
            **kwargs,
        )
    else:
        new_model = model
    return new_model
