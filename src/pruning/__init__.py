# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:07:26 2020

@author: Baptiste
"""

from .utils import getLinkLayerTrainableVariable, getInnerLayerTrainableVariables, getAllLayer, getInnerLayer, sparsityCallback, calcul_sparsity

from .L1heuristic import L1Heuristic
from .SNIPheuristic import SNIPHeuristic
from .Synflowheuristic import SynflowHeuristic
from .FPGMheuristic import  FPGMHeuristic

from .unstructuredpruning import UnstructuredPruning
from .structuredpruning import StructuredPruning

from .pruning_helper import prune