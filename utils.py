import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import mixed_precision
import numpy as np
import logging

try:
    # If using TensorFlow's mixed precision and distributed strategy
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
except ImportError:
    pass  # Mixed precision not supported in this TensorFlow version

logger = logging.getLogger(__name__)

def load_checkpoint(model, optimizer, config):
    logger.info(f"Resuming from {config['model_resume']}...")
    if config['model_resume'].startswith('http'):
        checkpoint_path = tf.keras.utils.get_file(fname=os.path.basename(config['model_resume']),
                                                  origin=config['model_resume'])
    else:
        checkpoint_path = config['model_resume']
    
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
        logger.info("Model weights loaded successfully from {}".format(checkpoint_path))
        
        # Optionally load optimizer state
        if 'load_optimizer_state' in config and config['load_optimizer_state']:
            optimizer.load_state_dict(torch.load(checkpoint_path + '_optimizer'))
            logger.info("Optimizer state loaded successfully.")
    else:
        logger.warning(f"No checkpoint found at {config['model_resume']} to resume from.")

def save_checkpoint(model, optimizer, config, epoch):
    model_save_path = os.path.join(config['output_path'], f'model_epoch_{epoch}.ckpt')
    optimizer_save_path = model_save_path + '_optimizer'
    model.save_weights(model_save_path)
    logger.info(f"Model weights saved to {model_save_path}")
    
    # Optionally save optimizer state
    if 'save_optimizer_state' in config and config['save_optimizer_state']:
        torch.save(optimizer.state_dict(), optimizer_save_path)
        logger.info(f"Optimizer state saved to {optimizer_save_path}")

def reduce_tensor(tensor, strategy):
    """
    Reduce tensor data across all nodes for distributed training
    """
    return strategy.reduce(tf.distribute.ReduceOp.SUM, tensor, axis=None) / strategy.num_replicas_in_sync

class AverageMeter:
    """Tracks and averages metrics"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def train_one_epoch(model, data_loader, optimizer, loss_fn, device, strategy):
    total_loss = AverageMeter()
    for step, (input, target) in enumerate(data_loader):
        with tf.GradientTape() as tape:
            predictions = model(input, training=True)
            loss = loss_fn(target, predictions)
            scaled_loss = optimizer.get_scaled_loss(loss) if isinstance(optimizer, mixed_precision.LossScaleOptimizer) else loss

        gradients = tape.gradient(scaled_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        reduced_loss = reduce_tensor(loss, strategy)
        total_loss.update(reduced_loss.numpy(), input.shape[0])
        if step % 100 == 0:
            logger.info(f'Step {step}, Loss: {total_loss.avg}')

    return total_loss.avg
