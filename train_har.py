import os
import sys
sys.path.append(os.path.abspath(''))
import random
import argparse
import numpy as np

from parser import add_training_args, add_rand_layer_args, add_pruning

def main(args):
    print("Random Seed: ", args.rand_seed)
    if args.rand_seed is not None:
        random.seed(args.rand_seed)
        tf.random.set_seed(args.rand_seed)
        np.random.seed(args.rand_seed)

    data_dir = "./data"

    domains = ['realworld', 'opportunity', 'hhar','pamap2']
    args.data_name = 'har'
    args.input_shape = (150,3)
    
    if args.multi_gpu:
        args.strategy = tf.distribute.MirroredStrategy()
        print ('Number of devices: {}'.format(args.strategy.num_replicas_in_sync))
        
    #Data augmentation
    args.n_classes = 14
    transform = [Cast(14)]
    
    if not args.inter:
        if args.source=="realworld":
            args.n_classes = 8
            transform = [Cast(8)]
    print("\n=========Preparing Data=========")
    
    print(args.source)
    #Load data and apply data transformations
    if args.inter:
        train_dataset = get_dataset(args.source, path=data_dir,inter=True,train=True, transform=transform)
        valid_datasets = {domain: get_dataset(domain, path=data_dir,inter=True,train=False, transform=transform) for domain in domains}
        #Filter elem of target dataset with label not in source dataset
        label_list,_ = tf.unique([tf.math.argmax(elem["label"]) for elem in train_dataset])
        valid_datasets = {d: valid_datasets[d].filter(Filter_label(label_list)) for d in valid_datasets}
        valid_datasets = {d: valid_datasets[d].apply(tf.data.experimental.assert_cardinality(len([i for i,elem in enumerate(valid_datasets[d])]))) for d in valid_datasets}

    else:
        train_dataset = get_dataset(args.source, path=data_dir,inter=False,train=True, transform=transform,source_localisation=args.source_localisation)
        valid_datasets = get_dataset(args.source, path=data_dir,inter=False,train=False, transform=transform,source_localisation=args.source_localisation)

    train_dataset = train_dataset.shuffle(2048).batch(args.batch_size)
    valid_datasets = {d: valid_datasets[d].shuffle(2048).batch(512) for d in valid_datasets}
    args.train_dataset_cardinality = train_dataset.cardinality().numpy()
    args.valid_datasets_cardinality = {d: valid_datasets[d].cardinality().numpy() for d in valid_datasets}
    
    #Gestion of dataset for Multi-GPU
    if args.multi_gpu:
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        train_dataset = train_dataset.with_options(options)
        valid_datasets = {d: valid_datasets[d].with_options(options) for d in valid_datasets}
        train_dataset = args.strategy.experimental_distribute_dataset(train_dataset)
        valid_datasets = {d: args.strategy.experimental_distribute_dataset(valid_datasets[d]) for d in valid_datasets}


    print("\n=========Building Model=========")
    #Creation of model
    if args.multi_gpu:
        with args.strategy.scope():
            net = get_network(args.net, args)
    else:
        net = get_network(args.net, args)
        
    net.summary()
    
    #Train model
    trainer = rct.RandCNN(args)
    trainer.train(net, train_dataset, valid_datasets)


if __name__ == '__main__':
    #Parser
    parser = argparse.ArgumentParser(description='Tensorflow HAR Training')
    add_training_args(parser)
    add_rand_layer_args(parser)
    add_pruning(parser)
    args = parser.parse_args()
    args.rand_conv_mode = "1D"
    
    #Environment settings
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'  
    if args.gpu_ids:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(elem) for elem in args.gpu_ids])
        
    import tensorflow as tf
    from src.datasets.data_helper import get_dataset
    from src.datasets.transforms import *
    from src.networks.network_helper import get_network
    import randconv_trainer as rct
        

    main(args)
