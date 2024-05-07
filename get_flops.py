import os
import time
import argparse
import datetime
import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from focalnet_tf.losses import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from focalnet_tf.utils import accuracy, AverageMeter
from focalnet_tf.data import CutmixMixupBlending
from focalnet_tf.config import get_config
from focalnet_tf.models import build_model
from focalnet_tf.data import build_dataloader
from focalnet_tf.schedulers import build_scheduler
from focalnet_tf.optimizers import build_optimizer
from focalnet_tf.logger import create_logger
from focalnet_tf.utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor
from focalnet_tf.layers import trunc_normal_init

from thop import profile, clever_format  # Note: THOP is not directly compatible with TensorFlow, requires adaptation

tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

def parse_option():
    parser = argparse.ArgumentParser('FocalNet training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=False, metavar="FILE", help='path to config file',
                        default='./configs/kinetics400/video-focalnet_tiny.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

_, config = parse_option()
config.defrost()
config.DATA.NUM_FRAMES = 8
config.freeze()
model = build_model(config)
model = model.to('gpu')  # Adaptation may be needed for TensorFlow
data = tf.random.normal((1, 8, 3, 224, 224))

# For profiling the model, you might need to adapt or find TensorFlow-compatible tools
macs, params = profile(model, inputs=(data, ))  # Adapt for TensorFlow
macs, _ = clever_format([macs, params], "%.3f")

print("gflops:", macs)
