__all__ = ['dataset_func_chain']

import functools

"""
Function to chain function for dataset transformations
"""

def dataset_func_chain(*functions):
    assert isinstance(*functions,list),"argument should be a list of function"
    if len(*functions)==0:
        return lambda x: x
    elif len(*functions)==1:
        return functions[0][0]
    else:
        def compose(f, g):
            return lambda x: g(f(x))
        return functools.reduce(compose, functions[0], lambda x: x)
