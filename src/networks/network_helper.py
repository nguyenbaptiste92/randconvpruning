__all__ = ['get_network']

from .digit_net import *
from .resnet import *
from .hhar_nn import *

network_map = {
    'digit': DigitNet,
    'pruned_digit' : PrunedDigitNet,
    'small_pruned_digit' : SmallPrunedDigitNet,
    'resnet20' : resnet_v1,
    'pruned_resnet20' : resnet_v1,
    'pruned_temportal_cnn1d':PrunedTemporalCNN1D,
}

"""
helper function in train_digits.py and train_har.py to create the network
arguments: -name (string): name of the network
           -args (list): arguments usefull (ex: input_shape, n_classes)
"""
    
def get_network(name, args):

    assert name in network_map
    if name=='digit':
        model = DigitNet(input_shape=args.input_shape,output_units=args.n_classes)
    elif name=='pruned_digit':
        model = PrunedDigitNet(input_shape=args.input_shape,output_units=args.n_classes)
    elif name=='small_pruned_digit':
        model = SmallPrunedDigitNet(input_shape=args.input_shape,output_units=args.n_classes)
    elif name=='resnet20':
        model = resnet_v1(args.input_shape, 20, num_classes=args.n_classes,prune=False)
    elif name=='pruned_resnet20':
        model = resnet_v1(args.input_shape, 20, num_classes=args.n_classes,prune=True)
    elif name=="pruned_temportal_cnn1d":
        model = PrunedTemporalCNN1D(args.input_shape, classes=args.n_classes)
    else:
        model = DigitNet(input_shape=args.input_shape,output_units=args.n_classes,pretrained=args.pretrained)
    return model
