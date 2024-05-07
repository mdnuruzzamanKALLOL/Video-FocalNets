import tensorflow as tf

def build_optimizer(config, model):
    """
    Build optimizer for TensorFlow, setting weight decay for non-normalization and non-bias parameters.
    """
    # Gather parameters with and without weight decay
    trainable_vars = model.trainable_variables
    has_decay = []
    no_decay = []
    
    for var in trainable_vars:
        if 'bias' in var.name or 'norm' in var.name:
            no_decay.append(var)
        else:
            has_decay.append(var)

    # Create weight decay using TensorFlow Addons if available
    try:
        import tensorflow_addons as tfa
        apply_decay = tfa.optimizers.weight_decay_optimizers.DecoupledWeightDecayExtension
    except ImportError:
        print("tensorflow_addons not installed, weight decay cannot be decoupled from optimizer.")
        apply_decay = None

    # Choose the optimizer
    opt_lower = config['train']['optimizer']['name'].lower()
    lr = config['train']['optimizer']['lr']
    weight_decay = config['train']['optimizer']['weight_decay']
    
    if opt_lower == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=config['train']['optimizer']['momentum'], nesterov=True)
    elif opt_lower == 'adamw':
        if apply_decay:
            optimizer = tfa.optimizers.AdamW(weight_decay=weight_decay, learning_rate=lr, epsilon=config['train']['optimizer']['eps'])
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # Apply weight decay to only non-bias and non-norm parameters if tensorflow_addons is available
    if apply_decay:
        optimizer = apply_decay(optimizer, weight_decay=weight_decay, include_in_weight_decay=has_decay, exclude_from_weight_decay=no_decay)

    return optimizer
