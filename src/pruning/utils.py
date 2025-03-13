__all__ = [
    "calcul_sparsity",
    "getAllLayer",
    "getInnerLayer",
    "getInnerLayerTrainableVariables",
    "getLinkLayerTrainableVariable",
    "sparsityCallback",
]

import tensorflow as tf

"""
Part to get inner layer and inner trainable variable
"""


def getLinkLayerTrainableVariable(model):
    dico = {}
    for layer in model.layers:
        result = getInnerLayerTrainableVariables(layer)
        for elem in result:
            for name in elem[1]:
                dico[name] = elem[0]
    return dico


def getInnerLayerTrainableVariables(layer):
    listinnerlayer = []
    test_layer = [
        inner_layer
        for inner_layer in layer._layers
        if (isinstance(inner_layer, (tf.keras.layers.Layer, tf.keras.Model)))
    ]
    if test_layer == []:
        return [
            [layer, [variable.name for variable in layer.trainable_variables]]
        ]
    for inner_layer in layer._layers:
        a = getInnerLayerTrainableVariables(inner_layer)
        listinnerlayer.extend(a)
    return listinnerlayer


def getAllLayer(model):
    listlayer = []
    for layer in model.layers:
        result = getInnerLayer(layer)
        listlayer.extend(result)
    return listlayer


def getInnerLayer(layer):
    listinnerlayer = []
    test_layer = [
        i
        for i in layer._layers
        if (isinstance(i, (tf.keras.layers.Layer, tf.keras.Model)))
    ]
    if test_layer == []:
        return [layer]
    for l in layer._layers:
        a = getInnerLayer(l)
        listinnerlayer.extend(a)
    return listinnerlayer


"""
Callback to display sparsity for unstructured pruning method
"""


class sparsityCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(sparsityCallback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        print("epochs end ", epoch)
        calcul_sparsity(self.model)


"""
Calcul and display sparsity of a model (pourcentage of zeros in tensor)
    Input:
        model : model to prune (class tf.keras.Model)
"""


def calcul_sparsity(model):
    # Pruning
    stats = []
    listlayer = getAllLayer(model)
    for layer in listlayer:
        if hasattr(layer, "kernel_mask"):
            stats.append(
                [
                    tf.math.count_nonzero(layer.kernel_mask).numpy(),
                    tf.size(layer.kernel_mask).numpy(),
                ],
            )
        if hasattr(layer, "bias_mask"):  # noqa: SIM102
            if layer.bias_mask is not None:
                stats.append(
                    [
                        tf.math.count_nonzero(layer.bias_mask).numpy(),
                        tf.size(layer.bias_mask).numpy(),
                    ],
                )

    if len(stats) == 0:
        print("Ce modele ne contient pas de couches prunees.")
    else:
        total = 0
        total_pruned = 0
        sparsity_layer = []
        for elem in stats:
            total += elem[1]
            total_pruned += elem[0]
            sparsity_layer.append(1 - elem[0] / elem[1])
        print(sparsity_layer)
        print(stats)
        print("proportion restante:", total_pruned / total)
