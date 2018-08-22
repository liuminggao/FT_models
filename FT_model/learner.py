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


def build_vgg16_feature_extraction(target_size, class_indices, freeze=True, is_classification=True):
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


def choice_build_fn(model_name):
    if model_name.lower() == 'xception':
        return build_xception_feature_extraction
    elif model_name.lower() == 'vgg16':
        return build_vgg16_feature_extraction
    else:
        raise RuntimeError()


class FTConvLearner:

    # TODO: Supoort regression problems

    def __init__(self, model_name, class_indices, target_size=(224, 224), optimizer=None, loss=None, metrics=None):
        """"
            model_name, str: 
                              which model to choice
            class_incices, dict: 
                              a dictionary for classes mapping 
                              like: {'dog': 0, 'cat': 1}
        """

        self.model_name = model_name
        self.target_size = target_size
        self.class_indices = class_indices

        self.build_fn = choice_build_fn(model_name)
        self.model, self.base_model = self.build_fn(target_size=target_size, class_indices=class_indices)

        self.optimizer = 'adam' if not optimizer else optimizer
        self.loss = 'categorical_crossentropy' if not loss else loss
        self.metrics = ['accuracy'] if not metrics else metrics

    def unfreeze(self):
        self.unfreeze_to(0)

    def unfreeze_to(self, n):
        """ network arch Top2Bottom unfreeze layer
        """
        for layer in self.base_model.layers[n:]:
            layer.trainable = True
        self._build_model()

    def _build_model(self):
        if self.model is None:
            raise RuntimeError()
        self.model.compile(self.optimizer, self.loss, self.metrics)

    def __str__(self):
        return 'Fine Tuning Model ({})'.format(self.model_name)

    def __repr__(self):
        return 'Fine Tuning Model ({})'.format(self.model_name)