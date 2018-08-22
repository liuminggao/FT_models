# -*- coding: utf-8 -*-

from keras import applications, layers, models


def build_xception_feature_extraction(target_size, class_indices, freeze=True):
    """Create feature extraction with Xception"""
    base_model = applications.xception.Xception(include_top=False, weights='imagenet',
                                                input_shape=(target_size[0], target_size[1], 3),
                                                pooling='avg')
    inputs = layers.Input(shape=(target_size[0], target_size[1], 3))
    x = inputs
    x = layers.Lambda(applications.xception.preprocess_input)(x)
    x = base_model(x)
    x = layers.Dense(len(class_indices), activation='softmax')(x)
    model = models.Model(inputs, x)
    if freeze:
        # freeze the weights already trained on ImageNet
        for layer in base_model.layers:
            layer.trainable = False
    return model, base_model


def build_vgg16_feature_extraction(target_size, class_indices, freeze=True):
    """Create feature extraction with VGG16"""
    base_model = applications.vgg16.VGG16(include_top=False, weights='imagenet',
                                          input_shape=(target_size[0], target_size[1], 3))
    inputs = layers.Input(shape=(target_size[0], target_size[1], 3))
    x = inputs
    x = layers.Lambda(applications.xception.preprocess_input)(x)
    x = base_model(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(len(class_indices), activation='softmax')(x)
    model = models.Model(inputs, x)
    if freeze:
        # freeze the weights already trained on ImageNet
        for layer in base_model.layers:
            layer.trainable = False
    return model, base_model


class FT_ConvLearner():

    def __init__(self, optimizer=None, loss=None, metrics=None):
        self._is_cls = True  #
        self.model = None
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def unfreeze(self):
        pass

    def unfreeze_to(self):
        pass

    def _build_model(self):
        if self.model is None:
            raise RuntimeError()
        self.model.compile(self.optimizer, self.loss, self.metrics)
