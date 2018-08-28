# -*- coding: utf-8 -*-

from keras import preprocessing
from keras.applications import xception

DEFAULT_TFMS = dict(rotation_range=20.0,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    zoom_range=0.2)


def get_batches(path, target_size, batch_size=32, shuffle=True, tfms=None):
    if tfms is not None:
        gen = preprocessing.image.ImageDataGenerator(**tfms, preprocessing_function=xception.preprocess_input)
    else:
        gen = preprocessing.image.ImageDataGenerator(tfms, preprocessing_function=xception.preprocess_input)
    batches = gen.flow_from_directory(path, target_size=target_size, batch_size=batch_size, shuffle=shuffle)
    return batches