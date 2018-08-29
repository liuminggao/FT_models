# -*- coding: utf-8 -*-

import tensorflow as tf
from keras import preprocessing
from keras.applications import xception

# FIXME 这个如何暴露给外边？
DEFAULT_TFMS = dict(rotation_range=20.0,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    zoom_range=0.2)


def get_batches(path, target_size, batch_size=32, shuffle=True, tfms=None):
    if tfms is not None:
        gen = preprocessing.image.ImageDataGenerator(**tfms, preprocessing_function=xception.preprocess_input)
    else:
        gen = preprocessing.image.ImageDataGenerator(preprocessing_function=xception.preprocess_input)
    batches = gen.flow_from_directory(path, target_size=target_size, batch_size=batch_size, shuffle=shuffle)
    return batches


def _export_estimator(est_model, name, shape, save_dir):
    feature_columns = [tf.feature_column.numeric_column(name, shape, normalizer_fn=xception.preprocess_input)]
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
    fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec=feature_spec)
    est_model.export_savedmodel(save_dir, serving_input_receiver_fn=fn)
    print('[INO] Estimator 模型保存成功.... [INFO]')