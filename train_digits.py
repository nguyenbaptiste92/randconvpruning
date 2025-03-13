import argparse
import os
import random
from pathlib import Path

import numpy as np

from parser import add_pruning, add_rand_layer_args, add_training_args


def main(args: argparse.Namespace) -> None:
    """
    Summary: Main function to train model on digits benchmark

    :param args: args from parser
    :type args: argparse.Namespace
    """
    data_dir = Path("./data")
    domains = ["mnist", "svhn", "usps"]
    args.n_classes = 10
    args.data_name = "digits"
    args.image_size = 32
    image_size = (32, 32)
    args.input_shape = (32, 32, 3)

    print("Random Seed: ", args.rand_seed)
    if args.rand_seed is not None:
        random.seed(args.rand_seed)
        tf.random.set_seed(args.rand_seed)
        np.random.seed(args.rand_seed)  # noqa: NPY002

    if args.multi_gpu:
        args.strategy = tf.distribute.MirroredStrategy()
        print(
            "Number of devices: {}".format(args.strategy.num_replicas_in_sync),
        )

    if args.source == "mnist10k":
        domains[0] = "mnist10k"

    # Data augmentation

    if args.multi_aug:
        train_transform = [
            Cast(args.n_classes),
            Rescale(255),
            Reshape(image_size),
            GreyToColor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ColorJitter(0.3, 0.3, 0.3, 0.3),
            RandomGrayscale(),
            DeNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    else:
        train_transform = [
            Cast(args.n_classes),
            Rescale(255),
            Reshape(image_size),
            GreyToColor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]

    test_transform = [
        Cast(args.n_classes),
        Rescale(255),
        Reshape(image_size),
        GreyToColor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    print("\n=========Preparing Data=========")

    # Load model and apply data transformation
    train_dataset = get_dataset(
        args.source,
        path=data_dir,
        train=True,
        transform=train_transform,
    )
    valid_datasets = {
        domain: get_dataset(
            domain,
            path=data_dir,
            train=False,
            transform=test_transform,
        )
        for domain in domains
    }

    train_dataset = train_dataset.shuffle(
        4096,
        reshuffle_each_iteration=True,
    ).batch(args.batch_size)
    valid_datasets = {
        d: valid_datasets[d]
        .shuffle(4096, reshuffle_each_iteration=True)
        .batch(args.batch_size)
        for d in valid_datasets
    }
    args.train_dataset_cardinality = train_dataset.cardinality().numpy()
    args.valid_datasets_cardinality = {
        d: valid_datasets[d].cardinality().numpy() for d in valid_datasets
    }

    # Gestion of dataset for Multi-GPU
    if args.multi_gpu:
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.DATA
        )
        train_dataset = train_dataset.with_options(options)
        valid_datasets = {
            d: valid_datasets[d].with_options(options) for d in valid_datasets
        }
        train_dataset = args.strategy.experimental_distribute_dataset(
            train_dataset,
        )
        valid_datasets = {
            d: args.strategy.experimental_distribute_dataset(valid_datasets[d])
            for d in valid_datasets
        }

    print("\n=========Building Model=========")
    # Creation of model
    if args.multi_gpu:
        with args.strategy.scope():
            net = get_network(args.net, args)
    else:
        net = get_network(args.net, args)

    # Train model
    trainer = rct.RandCNN(args)
    trainer.train(
        net,
        train_dataset,
        valid_datasets,
        data_mean=(0.5, 0.5, 0.5),
        data_std=((0.5, 0.5, 0.5)),
    )


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description="Tensorflow digits Training")
    add_training_args(parser)
    add_rand_layer_args(parser)
    add_pruning(parser)
    args = parser.parse_args()
    args.rand_conv_mode = "2D"

    # Environment settings
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    if args.gpu_ids:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(elem) for elem in args.gpu_ids],
        )

    import tensorflow as tf

    import randconv_trainer as rct
    from src.datasets.data_helper import get_dataset
    from src.datasets.transforms import (
        Cast,
        ColorJitter,
        DeNormalize,
        GreyToColor,
        Normalize,
        RandomGrayscale,
        Rescale,
        Reshape,
    )
    from src.networks.network_helper import get_network

    main(args)
