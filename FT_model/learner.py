# -*- coding: utf-8 -*-

import os

import numpy as np
from keras import applications, layers, models


def build_xception_feature_extraction(target_size, class_indices, freeze=True):
    """Create feature extraction with Xception"""
    base_model = applications.xception.Xception(include_top=False, weights='imagenet',
                                                input_shape=(target_size[0], target_size[1], 3),
                                                pooling='avg')
    inputs = layers.Input(shape=(target_size[0], target_size[1], 3))
    x = inputs
    # Lambda layer can make preprocess conviencely. unfortunately, it will get error when `load_model` stage
    # Solution: https://github.com/keras-team/keras/issues/8734#issuecomment-382602236
    # x = layers.Lambda(applications.xception.preprocess_input)(x)
    x = base_model(inputs)
    x = layers.Dense(len(class_indices), activation='softmax')(x)
    model = models.Model(inputs, x)
    if freeze:
        # freeze the weights already trained on ImageNet
        for layer in base_model.layers:
            layer.trainable = False
    return model, base_model


# def build_vgg16_feature_extraction(target_size, class_indices, freeze=True, is_classification=True):
#     """Create feature extraction with VGG16"""
#     base_model = applications.vgg16.VGG16(include_top=False, weights='imagenet',
#                                           input_shape=(target_size[0], target_size[1], 3))
#     inputs = layers.Input(shape=(target_size[0], target_size[1], 3))
#     x = inputs
#     x = layers.Lambda(applications.xception.preprocess_input)(x)
#     x = base_model(x)
#     x = layers.Flatten()(x)
#     x = layers.Dense(1024, activation='relu')(x)
#     x = layers.Dense(1024, activation='relu')(x)
#     x = layers.Dense(len(class_indices), activation='softmax')(x)
#     model = models.Model(inputs, x)
#     if freeze:
#         # freeze the weights already trained on ImageNet
#         for layer in base_model.layers:
#             layer.trainable = False
#     return model, base_model


def choice_build_fn(model_name):
    if model_name.lower() == 'xception':
        return build_xception_feature_extraction
    else:
        raise RuntimeError()


class FTConvLearner:

    # TODO: Supoort regression problems

    def __init__(self, class_indices, init=True, use_model_name='xception', target_size=(224, 224), optimizer=None, loss=None, metrics=None):
        """"
            model_name, str: 
                              which model to choice
            class_incices, dict: 
                              a dictionary for classes mapping 
                              like: {'dog': 0, 'cat': 1}
        """

        self.use_model_name = use_model_name
        self.target_size = target_size
        self.class_indices = class_indices

        self.optimizer = 'adam' if not optimizer else optimizer
        self.loss = 'categorical_crossentropy' if not loss else loss
        self.metrics = ['accuracy'] if not metrics else metrics

        self.model, self.base_model = None, None
        if init:
            self._init_model()
            self._build_model()

    def _init_model(self):
        self.build_fn = choice_build_fn(self.use_model_name)
        self.model, self.base_model = self.build_fn(target_size=self.target_size, class_indices=self.class_indices)

    def _build_model(self):
        if self.model is None:
            raise RuntimeError()
        self.model.compile(self.optimizer, self.loss, self.metrics)

    def unfreeze(self):
        self.unfreeze_to(0)

    def unfreeze_to(self, n):
        """ network arch Top2Bottom unfreeze layer
        """
        for layer in self.base_model.layers[n:]:
            layer.trainable = True
        self._build_model()

    def finetuning(self, batches, valid_batches, epochs):
        self.model.fit_generator(batches, steps_per_epoch=batches.n // batches.batch_size,
                                 validation_data=valid_batches, validation_steps=valid_batches.n // valid_batches.batch_size,
                                 epochs=epochs)

    def save(self, model_path, weight_path):
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(weight_path, exist_ok=True)
        with open(model_path, 'wt') as f:
            f.write(self.model.to_json())
        self.model.save_weights(weight_path)

    def load(self, model_path, weight_path):
        with open(model_path, 'rt') as f:
            json_string = f.read()
        self.model =  models.model_from_json(json_string)
        self.model.load_weights(weight_path)

    def predict_g(self, batches):
        y_prob = self.model.predict_genenrator(batches)
        y_pred = np.argmax(y_prob, axis=1)
        return y_prob, y_pred

    def __str__(self):
        return 'Fine Tuning Model ({})'.format(self.use_model_name)

    def __repr__(self):
        return 'Fine Tuning Model ({})'.format(self.use_model_name)