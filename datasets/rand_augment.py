import tensorflow as tf
import numpy as np
import random
import math
from PIL import Image, ImageEnhance, ImageOps

_FILL = (128, 128, 128)  # Default fill color for translate and rotate operations

def _interpolation(img, method=Image.BILINEAR):
    if method == 'bicubic':
        return Image.BICUBIC
    elif method == 'nearest':
        return Image.NEAREST
    else:
        return Image.BILINEAR

def shear_x(img, level, fill_color=_FILL):
    v = level / 10.0 * 0.3  # Shear level from -0.3 to 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0), fillcolor=fill_color)

def shear_y(img, level, fill_color=_FILL):
    v = level / 10.0 * 0.3  # Shear level from -0.3 to 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0), fillcolor=fill_color)

def translate_x(img, level, fill_color=_FILL):
    v = level / 10.0 * img.size[0] * 0.45  # Max translation of 45% of the image width
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0), fillcolor=fill_color)

def translate_y(img, level, fill_color=_FILL):
    v = level / 10.0 * img.size[1] * 0.45  # Max translation of 45% of the image height
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v), fillcolor=fill_color)

def rotate(img, level, fill_color=_FILL):
    degrees = level / 10.0 * 30  # Rotate between -30 and 30 degrees
    if random.random() > 0.5:
        degrees = -degrees
    return img.rotate(degrees, fillcolor=fill_color)

def auto_contrast(img):
    return ImageOps.autocontrast(img)

def invert(img):
    return ImageOps.invert(img)

def equalize(img):
    return ImageOps.equalize(img)

def solarize(img, level):
    thresh = 256 - level / 10.0 * 256
    return ImageOps.solarize(img, thresh)

def posterize(img, level):
    bits = int(level / 10.0 * 4)  # Reduce bits of each pixel to 4 - 0 bits
    return ImageOps.posterize(img, bits)

def contrast(img, level):
    factor = level / 10.0 * 1.8 + 0.1  # Contrast factor between 0.1 and 1.9
    return ImageEnhance.Contrast(img).enhance(factor)

def color(img, level):
    factor = level / 10.0 * 1.8 + 0.1  # Color factor between 0.1 and 1.9
    return ImageEnhance.Color(img).enhance(factor)

def brightness(img, level):
    factor = level / 10.0 * 1.8 + 0.1  # Brightness factor between 0.1 and 1.9
    return ImageEnhance.Brightness(img).enhance(factor)

def sharpness(img, level):
    factor = level / 10.0 * 1.8 + 0.1  # Sharpness factor between 0.1 and 1.9
    return ImageEnhance.Sharpness(img).enhance(factor)

def rand_augment_transform(num_layers=2, magnitude=10):
    """Generate a randomized augmentation policy."""
    available_transforms = [
        auto_contrast, equalize, invert, rotate, solarize, posterize, contrast,
        color, brightness, sharpness, shear_x, shear_y, translate_x, translate_y
    ]

    def augment(img):
        ops = random.sample(available_transforms, num_layers)
        for op in ops:
            img = op(img, magnitude)
        return img

    return augment
