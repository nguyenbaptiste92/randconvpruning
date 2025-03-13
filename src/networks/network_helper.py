__all__ = ["get_network"]

import argparse

import tensorflow as tf

from .digit_net import DigitNet, PrunedDigitNet, SmallPrunedDigitNet
from .hhar_nn import PrunedTemporalCNN1D
from .resnet import resnet_v1

network_map = {
    "digit": DigitNet,
    "pruned_digit": PrunedDigitNet,
    "small_pruned_digit": SmallPrunedDigitNet,
    "resnet20": resnet_v1,
    "pruned_resnet20": resnet_v1,
    "pruned_temportal_cnn1d": PrunedTemporalCNN1D,
}


def get_network(name: str, args: argparse.Namespace) -> tf.keras.Model:
    """
    Summary: helper function to create the network

    :param name: name of the network
    :type name: str
    :param args: args (ex: input_shape, n_classes)
    :type args: argparse.Namespace
    :return: _description_
    :rtype: tf.keras.Model
    """
    if name == "digit":
        model = DigitNet(
            input_shape=args.input_shape,
            output_units=args.n_classes,
        )
    elif name == "pruned_digit":
        model = PrunedDigitNet(
            input_shape=args.input_shape,
            output_units=args.n_classes,
        )
    elif name == "small_pruned_digit":
        model = SmallPrunedDigitNet(
            input_shape=args.input_shape,
            output_units=args.n_classes,
        )
    elif name == "resnet20":
        model = resnet_v1(
            args.input_shape,
            20,
            num_classes=args.n_classes,
            prune=False,
        )
    elif name == "pruned_resnet20":
        model = resnet_v1(
            args.input_shape,
            20,
            num_classes=args.n_classes,
            prune=True,
        )
    elif name == "pruned_temportal_cnn1d":
        model = PrunedTemporalCNN1D(
            args.input_shape,
            output_units=args.n_classes,
        )
    else:
        model = DigitNet(
            input_shape=args.input_shape,
            output_units=args.n_classes,
            pretrained=args.pretrained,
        )
    return model
