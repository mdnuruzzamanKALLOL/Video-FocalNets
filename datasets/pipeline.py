import tensorflow as tf
import numpy as np

def decode_video(filename):
    video_binary = tf.io.read_file(filename)
    video = tfio.experimental.ffmpeg.decode_video(video_binary)
    return video

def random_crop(video, size):
    return tf.image.random_crop(video, size=size + (video.shape[-1],))

def resize(video, size):
    return tf.image.resize(video, size)

def normalize(video, mean, std):
    return (video - mean) / std

class VideoDataset(tf.data.Dataset):
    """A TensorFlow dataset for loading and processing videos."""

    def __new__(cls, file_paths, labels, transforms):
        return tf.data.Dataset.from_tensor_slices((file_paths, labels)).map(
            cls._preprocess_function(transforms), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    @staticmethod
    def _preprocess_function(transforms):
        def preprocess(video_path, label):
            video = decode_video(video_path)
            for transform in transforms:
                video = transform(video)
            return video, label
        return preprocess

# Example usage
transforms = [
    lambda video: random_crop(video, (224, 224)),
    lambda video: resize(video, (224, 224)),
    lambda video: normalize(video, mean=127.5, std=127.5)
]
dataset = VideoDataset(file_paths=["path/to/video1.mp4", "path/to/video2.mp4"],
                       labels=[0, 1], transforms=transforms)

for video, label in dataset:
    print(video.shape, label)
