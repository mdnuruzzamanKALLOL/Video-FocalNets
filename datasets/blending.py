import tensorflow as tf

class BaseMiniBatchBlending(tf.Module):
    """Base class for mini-batch blending methods."""

    def __init__(self, num_classes, smoothing=0.):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing

    def one_hot(self, labels, smoothing=0.):
        """Apply label smoothing, converting labels to one-hot format."""
        off_value = smoothing / self.num_classes
        on_value = 1.0 - smoothing + off_value
        labels_one_hot = tf.one_hot(labels, depth=self.num_classes, on_value=on_value, off_value=off_value)
        return labels_one_hot

    def do_blending(self, imgs, labels):
        """To be implemented by subclasses."""
        raise NotImplementedError("Must be implemented in subclass.")

    def __call__(self, imgs, labels):
        labels_one_hot = self.one_hot(labels, smoothing=self.smoothing)
        mixed_imgs, mixed_labels = self.do_blending(imgs, labels_one_hot)
        return mixed_imgs, mixed_labels

class MixupBlending(BaseMiniBatchBlending):
    """Implementing Mixup for batch data."""

    def __init__(self, num_classes, alpha=0.2, smoothing=0.):
        super().__init__(num_classes, smoothing)
        self.alpha = alpha

    def do_blending(self, imgs, labels):
        """Apply Mixup on batch data."""
        batch_size = tf.shape(imgs)[0]
        indices = tf.random.shuffle(tf.range(batch_size))
        lam = tf.random.uniform([], minval=0.0, maxval=1.0)

        mixed_imgs = lam * imgs + (1 - lam) * tf.gather(imgs, indices)
        mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, indices)

        return mixed_imgs, mixed_labels

class CutmixBlending(BaseMiniBatchBlending):
    """Implementing Cutmix for batch data."""

    def __init__(self, num_classes, alpha=0.2, smoothing=0.):
        super().__init__(num_classes, smoothing)
        self.alpha = alpha

    def do_blending(self, imgs, labels):
        """Apply Cutmix on batch data."""
        batch_size = tf.shape(imgs)[0]
        indices = tf.random.shuffle(tf.range(batch_size))
        lam = tf.random.uniform([], minval=0.0, maxval=1.0)

        # Get random bbox
        cut_rat = tf.sqrt(1.0 - lam)
        cut_w = tf.cast(cut_rat * tf.cast(tf.shape(imgs)[2], tf.float32), tf.int32)
        cut_h = tf.cast(cut_rat * tf.cast(tf.shape(imgs)[1], tf.float32), tf.int32)
        cx = tf.random.uniform([], minval=0, maxval=tf.shape(imgs)[2], dtype=tf.int32)
        cy = tf.random.uniform([], minval=0, maxval=tf.shape(imgs)[1], dtype=tf.int32)

        bbx1 = tf.clip_by_value(cx - cut_w // 2, 0, tf.shape(imgs)[2])
        bby1 = tf.clip_by_value(cy - cut_h // 2, 0, tf.shape(imgs)[1])
        bbx2 = tf.clip_by_value(cx + cut_w // 2, 0, tf.shape(imgs)[2])
        bby2 = tf.clip_by_value(cy + cut_h // 2, 0, tf.shape(imgs)[1])

        mask = tf.logical_and(
            tf.logical_and(
                tf.range(tf.shape(imgs)[2], dtype=tf.int32) >= bbx1,
                tf.range(tf.shape(imgs)[2], dtype=tf.int32) < bbx2),
            tf.logical_and(
                tf.range(tf.shape(imgs)[1], dtype=tf.int32) >= bby1,
                tf.range(tf.shape(imgs)[1], dtype=tf.int32) < bby2))

        mask = tf.tile(tf.reshape(mask, (1, tf.shape(imgs)[1], tf.shape(imgs)[2], 1)), (batch_size, 1, 1, tf.shape(imgs)[3]))

        imgs = tf.where(mask, tf.gather(imgs, indices), imgs)
        # Adjust lambda according to the area of the patch
        lam = 1 - tf.reduce_sum(tf.cast(mask, tf.float32)) / tf.size(mask)
        mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, indices)

        return imgs, mixed_labels

# Example usage
num_classes = 10
batch_size = 32
img_height, img_width = 224, 224
num_channels = 3

# Create dummy data
imgs = tf.random.normal([batch_size, img_height, img_width, num_channels])
labels = tf.random.uniform([batch_size], minval=0, maxval=num_classes, dtype=tf.int32)

mixup = MixupBlending(num_classes=num_classes)
cutmix = CutmixBlending(num_classes=num_classes)

mixed_imgs_mixup, mixed_labels_mixup = mixup(imgs, labels)
mixed_imgs_cutmix, mixed_labels_cutmix = cutmix(imgs, labels)
