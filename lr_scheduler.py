import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

class CosineLRScheduler(LearningRateSchedule):
    def __init__(self, initial_learning_rate, min_lr, total_steps, warmup_learning_rate, warmup_steps):
        super(CosineLRScheduler, self).__init__()

        self.initial_learning_rate = initial_learning_rate
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        if step < self.warmup_steps:
            return self.warmup_learning_rate + step * (self.initial_learning_rate - self.warmup_learning_rate) / self.warmup_steps
        else:
            cos_decayed = 0.5 * (1 + tf.cos(tf.constant(math.pi) * (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
            decayed = (1 - self.min_lr) * cos_decayed + self.min_lr
            return self.initial_learning_rate * decayed

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "min_lr": self.min_lr,
            "total_steps": self.total_steps,
            "warmup_learning_rate": self.warmup_learning_rate,
            "warmup_steps": self.warmup_steps
        }

class LinearLRScheduler(LearningRateSchedule):
    def __init__(self, initial_learning_rate, final_learning_rate, total_steps, warmup_steps):
        super(LinearLRScheduler, self).__init__()

        self.initial_learning_rate = initial_learning_rate
        self.final_learning_rate = final_learning_rate
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        if step < self.warmup_steps:
            return self.initial_learning_rate + step * (self.final_learning_rate - self.initial_learning_rate) / self.warmup_steps
        else:
            linear_decay = (self.final_learning_rate - self.initial_learning_rate) / (self.total_steps - self.warmup_steps)
            return self.initial_learning_rate - (step - self.warmup_steps) * linear_decay

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "final_learning_rate": self.final_learning_rate,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps
        }

def build_scheduler(config, optimizer):
    if config.TRAIN.LR_SCHEDULER.NAME == 'cosine':
        return tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=config.TRAIN.BASE_LR,
            decay_steps=config.TRAIN.TOTAL_STEPS,
            alpha=config.TRAIN.MIN_LR
        )
    elif config.TRAIN.LR_SCHEDULER.NAME == 'linear':
        return LinearLRScheduler(
            initial_learning_rate=config.TRAIN.BASE_LR,
            final_learning_rate=config.TRAIN.MIN_LR,
            total_steps=config.TRAIN.TOTAL_STEPS,
            warmup_steps=config.TRAIN.WARMUP_STEPS
        )
    else:
        raise ValueError("Unsupported LR Scheduler: {}".format(config.TRAIN.LR_SCHEDULER.NAME))
