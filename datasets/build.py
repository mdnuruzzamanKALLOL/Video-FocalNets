import tensorflow as tf
import numpy as np
import pandas as pd
import os

def load_video_frames(video_path, num_frames):
    """Load video frames and uniformly sample them across the video."""
    video = tf.io.read_file(video_path)
    video = tf.io.decode_video(video)
    total_frames = tf.shape(video)[0]
    stride = tf.maximum(1, tf.cast(total_frames / num_frames, tf.int32))
    indices = tf.range(start=0, limit=total_frames, delta=stride)
    sampled_frames = tf.gather(video, indices[:num_frames])
    return sampled_frames

def preprocess_video(frames, resize_shape=(224, 224), augment=False):
    """Preprocess video frames with resizing and optional augmentation."""
    frames = tf.image.resize(frames, resize_shape)
    if augment:
        frames = tf.image.random_flip_left_right(frames)
        frames = tf.image.random_brightness(frames, max_delta=0.3)
        frames = tf.image.random_contrast(frames, lower=0.8, upper=1.2)
    frames = (frames - 127.5) / 127.5  # Normalize to [-1, 1]
    return frames

def prepare_labels(label, num_classes):
    """Convert label index to one-hot encoded vector."""
    return tf.one_hot(label, depth=num_classes)

def load_and_preprocess_from_path_label(data_dir, label_list, video_path, label):
    """Load video and labels given a video file path and label."""
    video = load_video_frames(os.path.join(data_dir, video_path), num_frames=16)
    video = preprocess_video(video, augment=True)
    label = prepare_labels(label, num_classes=len(label_list))
    return video, label

def create_dataset(data_dir, ann_file, label_list, batch_size, repeat_count=1):
    """Create a TensorFlow dataset from annotation file and label list."""
    df = pd.read_csv(ann_file)
    video_paths = df['video_path'].values
    labels = df['label'].values
    dataset = tf.data.Dataset.from_tensor_slices((video_paths, labels))
    dataset = dataset.map(lambda x, y: load_and_preprocess_from_path_label(data_dir, label_list, x, y))
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).repeat(repeat_count)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

# Usage
data_dir = '/path/to/videos'
ann_file = '/path/to/annotation.csv'
label_list = '/path/to/label_list.csv'
batch_size = 32

train_dataset = create_dataset(data_dir, ann_file, label_list, batch_size)
for video_batch, label_batch in train_dataset:
    print(video_batch.shape, label_batch.shape)  # Output shape of batch
