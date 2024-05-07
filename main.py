import os
import time
import argparse
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from timm_tf import create_model, Mixup, LabelSmoothingCrossEntropy
from data import build_dataset
from scheduler import build_lr_scheduler
from optimizer import build_optimizer
from utils import load_checkpoint, save_checkpoint, create_logger, reduce_tensor
from config import get_config

def parse_option():
    parser = argparse.ArgumentParser('FocalNet training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, help='path to config file')
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, help='root of output folder')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    args = parser.parse_args()
    config = get_config(args.cfg, args.opts)
    return args, config

def main(config):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = create_model(config.model.name, pretrained=config.model.pretrained)
        optimizer = build_optimizer(model, config)
        if config.amp_opt_level != 'O0':
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_policy(policy)
            optimizer = mixed_precision.LossScaleOptimizer(optimizer, dynamic=True)

        if config.resume:
            model.load_weights(config.resume)

        train_dataset = build_dataset(config, training=True)
        val_dataset = build_dataset(config, training=False)

        train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
        val_dist_dataset = strategy.experimental_distribute_dataset(val_dataset)

        loss_fn = LabelSmoothingCrossEntropy()
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

        if config.train.use_mixup:
            mixup = Mixup(mixup_alpha=config.train.mixup_alpha, cutmix_alpha=config.train.cutmix_alpha)

    @tf.function
    def train_step(inputs):
        images, labels = inputs
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_fn(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_accuracy.update_state(labels, predictions)
        return loss

    @tf.function
    def test_step(inputs):
        images, labels = inputs
        predictions = model(images, training=False)
        t_loss = loss_fn(labels, predictions)
        val_accuracy.update_state(labels, predictions)
        return t_loss

    for epoch in range(config.train.epochs):
        total_loss = 0.0
        num_batches = 0
        for x_batch_train in train_dist_dataset:
            total_loss += strategy.run(train_step, args=(x_batch_train,))
            num_batches += 1
        train_loss = total_loss / num_batches

        for x_batch_val in val_dist_dataset:
            strategy.run(test_step, args=(x_batch_val,))

        template = ("Epoch {}, Loss: {}, Accuracy: {}, Val Loss: {}, Val Accuracy: {}")
        print(template.format(epoch+1,
                              train_loss,
                              train_accuracy.result()*100,
                              val_loss,
                              val_accuracy.result()*100))

        # Reset the metrics for the next epoch
        train_accuracy.reset_states()
        val_accuracy.reset_states()

if __name__ == '__main__':
    args, config = parse_option()
    main(config)
