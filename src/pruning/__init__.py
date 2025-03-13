# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:07:26 2020

@author: Baptiste
"""

from .fpgm_heuristic import fpgm_heuristic
from .l1_heuristic import l1_heuristic
from .pruning_helper import prune
from .snip_heuristic import snip_heuristic
from .structuredpruning import structuredpruning
from .synflow_heuristic import synflow_heuristic
from .unstructuredpruning import unstructuredpruning
from .utils import (
    calcul_sparsity,
    getAllLayer,
    getInnerLayer,
    getInnerLayerTrainableVariables,
    getLinkLayerTrainableVariable,
    sparsityCallback,
)
