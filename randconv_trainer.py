import os
import sys

sys.path.append(os.path.abspath(""))

import time
from pathlib import Path
from random import randint
from typing import Dict, Tuple

import tensorflow as tf

from parser import *
from src.datasets.transforms import *
from src.layers.rand_conv import *
from src.pruning.pruning_helper import prune
from src.pruning.utils import calcul_sparsity


def get_random_module(args, data_mean, data_std):
    return RandConvModule(
        filters=args.channel_size,
        kernel_sizes=args.kernel_size,
        mode=args.rand_conv_mode,
        mixing=args.mixing,
        identity_prob=args.identity_prob,
        rand_bias=args.rand_bias,
        distribution=args.distribution,
        data_mean=data_mean,
        data_std=data_std,
        clamp_output=args.clamp_output,
    )


class RandCNN:
    def __init__(self, args):
        self.args = args

        if len(self.args.n_iter) > 0:
            self.args.n_epoch = [
                i // self.args.val_iter for i in self.args.n_iter
            ]

        self.exp_name = get_exp_name(self.args)

        print(
            "{} experiments with {} as source domain {}".format(
                self.args.data_name,
                self.args.source,
                self.exp_name,
            ),
        )

        # Creation of log file
        self.set_path()
        Path(os.getcwd() + os.sep + self.log_folder).mkdir(
            parents=True,
            exist_ok=True,
        )
        with open(
            os.getcwd() + os.sep + self.log_folder + os.sep + "parameter.txt",
            "w+",
        ) as file_object:
            file_object.write("Parameter of experiment.\n")
            file_object.write(self.exp_name)

        # Declaration of loss and metrics
        # Multi-GPU
        if self.args.multi_gpu:
            self.strategy = self.args.strategy
            with self.strategy.scope():
                self.criterion = tf.keras.losses.CategoricalCrossentropy(
                    reduction=tf.keras.losses.Reduction.NONE,
                )
                if self.args.consistency_loss:
                    self.invariant_criterion = tf.keras.losses.KLDivergence(
                        reduction=tf.keras.losses.Reduction.NONE,
                    )
                else:
                    self.invariant_criterion = None
                self.metrics = {}
                for metric in self.args.metric:
                    if metric == "accuracy":
                        self.metrics["accuracy"] = (
                            tf.keras.metrics.CategoricalAccuracy()
                        )
                    elif metric == "f1score":
                        self.metrics["f1score"] = tf.keras.metrics.F1Score(
                            average="macro",
                        )

                self.pred_loss_metric = tf.keras.metrics.Mean()
                self.consticency_loss_metric = tf.keras.metrics.Mean()
        # Mono-GPU
        else:
            self.criterion = tf.keras.losses.CategoricalCrossentropy()
            if self.args.consistency_loss:
                self.invariant_criterion = tf.keras.losses.KLDivergence()
            else:
                self.invariant_criterion = None
            self.metrics = {}
            for metric in self.args.metric:
                if metric == "accuracy":
                    self.metrics["accuracy"] = (
                        tf.keras.metrics.CategoricalAccuracy()
                    )
                elif metric == "f1score":
                    self.metrics["f1score"] = tf.keras.metrics.F1Score(
                        average="macro",
                    )
            self.pred_loss_metric = tf.keras.metrics.Mean()
            self.consticency_loss_metric = tf.keras.metrics.Mean()

        # Declaration of pruning parameter
        self.pruning = self.args.pruning
        if self.pruning:
            self.pruning_schedule = self.args.pruning_schedule
            self.rewinding = self.args.rewinding
            if self.rewinding:
                self.rewinding_epoch = self.args.rewinding_epoch

        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

    # Function to select log folder path: if the path already exist, randomization is used to create another path
    def set_path(self):
        self.log_folder = os.path.join(
            "logs",
            self.args.data_name,
            self.args.exp_name,
            str(self.args.job_id),
        )
        while os.path.exists(self.log_folder):
            self.log_folder = self.log_folder + "_" + str(randint(0, 1000))
        self.model_folder = os.path.join(
            "saved_model",
            self.args.data_name,
            self.args.exp_name,
            str(self.args.job_id),
        )
        while os.path.exists(self.model_folder):
            self.model_folder = self.model_folder + "_" + str(randint(0, 1000))

    # Function to create the optimizer depending of the arguments given by the user
    def set_optimizer_and_scheduler(
        self,
        learning_rate,
        SGD=False,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=False,
        scheduler_name="",
        step_size=20,
        gamma=0.1,
        milestones=(10, 20),
        n_epoch=30,
    ):
        if scheduler_name == "StepLR":
            scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
                learning_rate,
                decay_steps=self.n_steps_per_epoch * step_size,
                decay_rate=gamma,
                staircase=True,
            )
        elif scheduler_name == "MultiStepLR":
            values = [learning_rate] + [
                learning_rate * (gamma ** (i + 1))
                for i in range(len(milestones))
            ]
            milestones = [
                miles * self.n_steps_per_epoch for miles in milestones
            ]
            scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                boundaries=milestones,
                values=values,
            )
        elif scheduler_name == "CosLR":
            scheduler = tf.keras.optimizers.schedules.CosineDecay(
                learning_rate,
                self.n_steps_per_epoch * n_epoch,
            )
        elif not scheduler_name:
            scheduler = None
        else:
            raise NotImplementedError

        # only update non-random layers
        if SGD:
            print("Using SGD optimizer")
            if scheduler is None:
                optimizer = tf.keras.optimizers.SGD(
                    learning_rate=learning_rate,
                    momentum=momentum,
                    nesterov=nesterov,
                    weight_decay=weight_decay,
                )
            else:
                optimizer = tf.keras.optimizers.SGD(
                    learning_rate=scheduler,
                    momentum=momentum,
                    nesterov=nesterov,
                    weight_decay=weight_decay,
                )
        else:
            print("Using Adam optimizer")
            if scheduler is None:
                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=learning_rate,
                )
            else:
                optimizer = tf.keras.optimizers.Adam(learning_rate=scheduler)

        return optimizer, scheduler

    # Training function
    def train(
        self,
        model,
        train_dataset,
        valid_datasets,
        data_mean=None,
        data_std=None,
    ):
        """
        Training a classfication CNN with random layers
        :param net: base model
        :param train_dataset: dataset used for training
        :param valid_datasets: dict of all the datasets use for validation
        :param data_mean: mean of dataset (a vector of 3 for color images)
        :param data_std: std of dataset (a vector of 3 for color images)
        :return:
        """
        self.data_mean = data_mean
        self.data_std = data_std
        self.best_metric = 0  # best valid accuracy
        self.best_target_metric = 0  # best valid accuracy on target domain
        self.current_metric = 0
        start_epoch = 0  # start from epoch 0 or last checkpoint epoch

        self.model = model

        # get the random conv layers and trainable parameters : Need to add the part which can select only classifier parameters

        if self.args.rand_conv:
            print("\n=========Set Up Rand layers=========")
            self.rand_module = get_random_module(self.args, data_mean, data_std)
        else:
            self.rand_module = None

        if len(self.args.n_iter) > 0:
            self.n_steps_per_epoch = self.args.val_iter
        else:
            self.n_steps_per_epoch = train_dataset.cardinality().numpy()

        # Set optimizer
        if self.args.multi_gpu:
            with self.strategy.scope():
                self.optimizer, self.scheduler = (
                    self.set_optimizer_and_scheduler(
                        learning_rate=self.args.lr,
                        SGD=self.args.SGD,
                        momentum=self.args.momentum,
                        weight_decay=self.args.weight_decay,
                        nesterov=self.args.nesterov,
                        scheduler_name=self.args.scheduler,
                        step_size=self.args.step_size,
                        gamma=self.args.gamma,
                        milestones=self.args.milestones,
                        n_epoch=self.args.n_epoch,
                    )
                )
        else:
            self.optimizer, self.scheduler = self.set_optimizer_and_scheduler(
                learning_rate=self.args.lr,
                SGD=self.args.SGD,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
                nesterov=self.args.nesterov,
                scheduler_name=self.args.scheduler,
                step_size=self.args.step_size,
                gamma=self.args.gamma,
                milestones=self.args.milestones,
                n_epoch=self.args.n_epoch,
            )

        start = time.time()
        previous_sparsity = 0.0

        # Train model
        # No pruning
        if not self.pruning:
            print("\n=========Training with rand layers=========")
            self.writer = tf.summary.create_file_writer(
                os.path.join(self.log_folder, "nopruning"),
            )
            for epoch in range(start_epoch, self.args.n_epoch[0]):
                self.train_epoch(epoch + 1, self.args.n_epoch[0], train_dataset)
                self.validate_epoch(
                    epoch + 1,
                    self.args.n_epoch[0],
                    valid_datasets,
                    n_eval=self.args.n_val if self.args.val_with_rand else 1,
                )
        # Pruning
        # Finetuning or Learning Rate Rewinding
        elif not self.rewinding:
            for i in range(len(self.pruning_schedule)):
                print(
                    "\n=========Pruning model with sparsity of {}%.=========".format(
                        self.pruning_schedule[i],
                    ),
                )
                self.model = prune(
                    self.model,
                    self.pruning_schedule[i],
                    previous_sparsity,
                    train_dataset,
                    self,
                )
                previous_sparsity = self.pruning_schedule[i]
                calcul_sparsity(self.model)
                self.writer = tf.summary.create_file_writer(
                    os.path.join(
                        self.log_folder,
                        "pruning_sparsity{}".format(self.pruning_schedule[i]),
                    ),
                )
                print("\n=========Training with rand layers=========")
                if self.args.finetuning and i == 1:
                    # learning_rate = self.optimizer._decayed_lr(tf.float32)
                    if self.args.multi_gpu:
                        with self.strategy.scope():
                            self.optimizer, self.scheduler = (
                                self.set_optimizer_and_scheduler(
                                    learning_rate=learning_rate,
                                    SGD=self.args.SGD,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay,
                                    nesterov=self.args.nesterov,
                                    scheduler_name="",
                                )
                            )
                    else:
                        self.optimizer, self.scheduler = (
                            self.set_optimizer_and_scheduler(
                                learning_rate=learning_rate,
                                SGD=self.args.SGD,
                                momentum=self.args.momentum,
                                weight_decay=self.args.weight_decay,
                                nesterov=self.args.nesterov,
                                scheduler_name="",
                            )
                        )
                else:
                    for var in self.optimizer.variables():
                        var.assign(tf.zeros_like(var))
                self.validate_epoch(
                    0,
                    self.args.n_epoch[i],
                    valid_datasets,
                    n_eval=self.args.n_val if self.args.val_with_rand else 1,
                )
                for epoch in range(start_epoch, self.args.n_epoch[i]):
                    epoch_start = time.time()
                    self.train_epoch(
                        epoch + 1,
                        self.args.n_epoch[i],
                        train_dataset,
                    )
                    self.validate_epoch(
                        epoch + 1,
                        self.args.n_epoch[i],
                        valid_datasets,
                        n_eval=self.args.n_val
                        if self.args.val_with_rand
                        else 1,
                    )
                    epoch_end = time.time()
                    print(
                        "Epoch_Training_time : ",
                        epoch_end - epoch_start,
                        " sec.",
                    )

        # Weight rewinding
        else:
            # Begin training
            self.writer = tf.summary.create_file_writer(
                os.path.join(self.log_folder, "pruning_sparsity{}".format(0)),
            )
            for epoch in range(start_epoch, self.rewinding_epoch):
                epoch_start = time.time()
                self.train_epoch(epoch + 1, self.args.n_epoch[0], train_dataset)
                self.validate_epoch(
                    epoch + 1,
                    self.args.n_epoch[0],
                    valid_datasets,
                    n_eval=self.args.n_val if self.args.val_with_rand else 1,
                )
                epoch_end = time.time()
                print(
                    "Epoch_Training_time : ",
                    epoch_end - epoch_start,
                    " sec.",
                )

            # Save weight
            print("Save for rewinding.")
            self.rewinding_weights = [
                tf.identity(variable)
                for variable in self.model.trainable_variables
            ]
            self.rewinding_optimizer_var = [
                tf.identity(variable) for variable in self.optimizer.variables
            ]

            # End Initial Training
            for epoch in range(self.rewinding_epoch, self.args.n_epoch[0]):
                epoch_start = time.time()
                self.train_epoch(epoch + 1, self.args.n_epoch[0], train_dataset)
                self.validate_epoch(
                    epoch + 1,
                    self.args.n_epoch[0],
                    valid_datasets,
                    n_eval=self.args.n_val if self.args.val_with_rand else 1,
                )
                epoch_end = time.time()
                print(
                    "Epoch_Training_time : ",
                    epoch_end - epoch_start,
                    " sec.",
                )
            # Pruning
            for i in range(len(self.pruning_schedule)):
                print(
                    "\n=========Rewind pruning model with sparsity of {}%.=========".format(
                        self.pruning_schedule[i],
                    ),
                )
                self.model = prune(
                    self.model,
                    self.pruning_schedule[i],
                    previous_sparsity,
                    train_dataset,
                    self,
                )
                for variable, rewind_weight in zip(
                    self.model.trainable_variables,
                    self.rewinding_weights,
                    strict=False,
                ):
                    variable.assign(rewind_weight)
                for var, rewinding_var in zip(
                    self.optimizer.variables,
                    self.rewinding_optimizer_var,
                    strict=False,
                ):
                    var.assign(rewinding_var)

                previous_sparsity = self.pruning_schedule[i]
                calcul_sparsity(self.model)

                print("\n=========Training with rand layers=========")
                self.writer = tf.summary.create_file_writer(
                    os.path.join(
                        self.log_folder,
                        "pruning_sparsity{}".format(self.pruning_schedule[i]),
                    ),
                )
                self.validate_epoch(
                    0,
                    self.args.n_epoch[0],
                    valid_datasets,
                    n_eval=self.args.n_val if self.args.val_with_rand else 1,
                )
                for epoch in range(self.rewinding_epoch, self.args.n_epoch[0]):
                    epoch_start = time.time()
                    self.train_epoch(
                        epoch + 1,
                        self.args.n_epoch[0],
                        train_dataset,
                    )
                    self.validate_epoch(
                        epoch + 1,
                        self.args.n_epoch[0],
                        valid_datasets,
                        n_eval=self.args.n_val
                        if self.args.val_with_rand
                        else 1,
                    )
                    epoch_end = time.time()
                    print(
                        "Epoch_Training_time : ",
                        epoch_end - epoch_start,
                        " sec.",
                    )

        end = time.time()
        print("Training_time : ", end - start, " sec.")

        # Save model
        if self.args.save_model:
            model_path = os.path.join(self.model_folder, "model")
            print(model_path)
            self.model.save_weights(model_path)

    #################### TRAIN FUNCTIONS #######################################

    def train_step(
        self,
        data: Dict[str, tf.Tensor],
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Summary: train step with rand conv training

        :param data: dict with "image" and "label" key
        :type data: Dict[str, tf.Tensor]
        :return: cross entropy loss, invariant loss
        :rtype: Tuple[tf.Tensor, tf.Tensor]
        """
        with tf.GradientTape() as tape:
            pred_loss = 0.0
            inv_loss = 0.0

            inputs = data["image"]
            target = data["label"]

            # Augment and randomize
            if not (
                self.rand_module is None
                or (self.args.consistency_loss and self.args.augmix)
            ):
                inputs = self.rand_module(inputs)

            # Compute pred loss
            outputs = self.model(inputs)["label"]
            if self.args.multi_gpu:
                pred_loss += tf.reduce_sum(self.criterion(target, outputs)) * (
                    1.0 / self.args.batch_size
                )
            else:
                pred_loss += self.criterion(target, outputs)

            # Compute invariant loss
            if (
                self.rand_module is not None
                and self.invariant_criterion is not None
            ):
                # self.rand_module.randomize()//Randomization include
                # in the call function
                inputs1 = self.rand_module(inputs)
                inputs2 = self.rand_module(inputs)

                outputs1 = self.model(inputs1)["label"]
                outputs2 = self.model(inputs2)["label"]

                if self.args.consistency_loss:
                    p_mixture = (
                        tf.math.add_n([outputs, outputs1, outputs2]) / 3.0
                    )
                    if self.args.multi_gpu:
                        inv_loss += (
                            tf.math.add_n(
                                [
                                    tf.reduce_sum(
                                        self.invariant_criterion(
                                            p_mixture,
                                            outputs,
                                        ),
                                    )
                                    * (1.0 / self.args.batch_size),
                                    tf.reduce_sum(
                                        self.invariant_criterion(
                                            p_mixture,
                                            outputs1,
                                        ),
                                    )
                                    * (1.0 / self.args.batch_size),
                                    tf.reduce_sum(
                                        self.invariant_criterion(
                                            p_mixture,
                                            outputs2,
                                        ),
                                    )
                                    * (1.0 / self.args.batch_size),
                                ],
                            )
                            / 3.0
                        )
                    else:
                        inv_loss += (
                            tf.math.add_n(
                                [
                                    self.invariant_criterion(
                                        p_mixture,
                                        outputs,
                                    ),
                                    self.invariant_criterion(
                                        p_mixture,
                                        outputs1,
                                    ),
                                    self.invariant_criterion(
                                        p_mixture,
                                        outputs2,
                                    ),
                                ],
                            )
                            / 3.0
                        )

            # Compute gradients and update model
            total_loss = pred_loss + self.args.consistency_loss_w * inv_loss

        grad = tape.gradient(
            total_loss,
            self.model.trainable_variables,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )
        self.optimizer.apply_gradients(
            zip(grad, self.model.trainable_variables, strict=False),
        )

        # Update Metric
        for metric in self.metrics.values():
            metric.update_state(target, outputs)
        self.pred_loss_metric.update_state(pred_loss)
        self.consticency_loss_metric.update_state(inv_loss)

        return pred_loss, inv_loss

    # MultiGPU
    @tf.function
    def distributed_train_step(
        self,
        data: Dict[str, tf.Tensor],
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Summary: distributed train step with rand conv training

        :param data: dict with "image" and "label" key
        :type data: Dict[str, tf.Tensor]
        :return: cross entropy loss, invariant loss
        :rtype: Tuple[tf.Tensor, tf.Tensor]
        """
        per_replica_pred_loss, per_replica_inv_loss = self.strategy.run(
            self.train_step,
            args=(data,),
        )
        return self.strategy.reduce(
            "SUM",
            per_replica_pred_loss,
            axis=None,
        ), self.strategy.reduce("SUM", per_replica_inv_loss, axis=None)

    def train_epoch(self, epoch, max_epoch, train_dataset):
        # Reinitialize metrics for epoch
        for metric in self.metrics.values():
            metric.reset_state()
        self.pred_loss_metric.reset_state()
        self.consticency_loss_metric.reset_state()

        # update summary
        """with self.writer.as_default():
            tf.summary.scalar(
                "Learning rate",
                self.optimizer._decayed_lr(tf.float32),
                step=epoch,
            )"""

        # Declare Progbar
        print(
            "\nTraining epoch {}/{} with {}".format(
                epoch,
                max_epoch,
                self.args.source,
            ),
        )
        progBar = tf.keras.utils.Progbar(
            self.n_steps_per_epoch,
            stateful_metrics=["pred_loss", "consistency_loss"]
            + list(self.metrics.keys()),
            verbose=self.args.verbose,
        )

        # Training Loop
        for i, elem in enumerate(train_dataset):
            if i >= self.n_steps_per_epoch:
                break
            if self.args.multi_gpu:
                pred_loss, inv_loss = self.distributed_train_step(elem)
            else:
                pred_loss, inv_loss = self.train_step(elem)
            values = [
                ("pred_loss", pred_loss),
                ("consistency_loss", inv_loss),
            ] + [(key, metric.result()) for key, metric in self.metrics.items()]
            progBar.update(i + 1, values=values)
            # Update the global step
            self.global_step.assign_add(1)

        # Write value in summary for epoch
        with self.writer.as_default():
            tf.summary.scalar(
                "Pred_loss_epoch",
                self.pred_loss_metric.result(),
                step=epoch,
            )
            tf.summary.scalar(
                "Consistency_loss_epoch",
                self.consticency_loss_metric.result(),
                step=epoch,
            )
            for key, metric in self.metrics.items():
                tf.summary.scalar(
                    "Training_" + key,
                    metric.result(),
                    step=epoch,
                )

        if not self.args.verbose:
            liste_print = [
                ("Training_" + key, metric.result())
                for key, metric in self.metrics.items()
            ]
            print(
                "Pred_loss_epoch:",
                self.pred_loss_metric.result(),
                " Consistency_loss_epoch:",
                self.consticency_loss_metric.result(),
                liste_print,
            )

    #################### TEST FUNCTIONS ###############################################################################################################

    def validate_step(self, elem):
        inputs = elem["image"]
        target = elem["label"]

        if self.rand_module is not None and self.args.val_with_rand:
            inputs = self.rand_module(inputs)

        outputs = self.model(inputs, training=False)["label"]
        for metric in self.metrics.values():
            metric.update_state(target, outputs)

    # MultiGPU
    @tf.function
    def distributed_validate_step(self, elem):
        self.strategy.run(self.validate_step, args=(elem,))

    def validate_epoch(self, epoch, maxepoch, valid_datasets, n_eval):
        assert isinstance(valid_datasets, dict)

        self.model.compile(
            loss=self.criterion,
            metrics=list(self.metrics.values()),
        )
        for name, dataset in valid_datasets.items():
            print(
                "\nEvaluation epoch {}/{} on {}".format(epoch, maxepoch, name),
            )

            progBar = tf.keras.utils.Progbar(
                self.args.valid_datasets_cardinality[name],
                stateful_metrics=list(self.metrics.keys()),
                verbose=self.args.verbose,
            )
            for metric in self.metrics.values():
                metric.reset_state()

            for i, elem in enumerate(dataset):
                for j in range(n_eval):
                    if self.args.multi_gpu:
                        self.distributed_validate_step(elem)
                    else:
                        self.validate_step(elem)

                values = [
                    (key, metric.result())
                    for key, metric in self.metrics.items()
                ]
                progBar.update(i + 1, values=values)

            with self.writer.as_default():
                for key, metric in self.metrics.items():
                    tf.summary.scalar(
                        name + "_" + "Validation_" + key,
                        metric.result(),
                        step=epoch,
                    )
            if not self.args.verbose:
                liste_print = [
                    (name + "_Validation_" + key, metric.result())
                    for key, metric in self.metrics.items()
                ]
                print(liste_print)
