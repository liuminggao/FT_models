# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
from keras import applications, layers, models


def build_model(model):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


def pretrained_model(arch, input_shape=(224, 224, 3)):
    """
         返回对应在 ImageNet 训练好的网络模型

         args:
             arch: str, 模型名字, 
                        eg: 'xception'
             input_shape, tuple, 模型输入维度, 
                        eg: (224, 224, 3)

         return:
              model, keras model, 一个没有被freeze的网络模型

    """
    arch = arch.lower()
    arch_set = {'xception', 'vgg16', 'vgg19', 'resnet50', 'inception_v3'}
    if arch == 'xception':
        model = applications.Xception(include_top=True, weights='imagenet', input_shape=input_shape)
    elif arch == 'vgg16':
        model = applications.VGG16(include_top=True, weights='imagenet', input_shape=input_shape)
    elif arch == 'vgg19':
        model = applications.VGG19(include_top=True, weights='imagenet', input_shape=input_shape)
    elif arch == 'resnet50':
        model = applications.ResNet50(include_top=True, weights='imagenet', input_shape=input_shape)
    elif arch == 'inception_v3':
        model = applications.InceptionV3(include_top=True, weights='imagenet', input_shape=input_shape)
    else:
        raise ValueError('使用以下网络结构 \n{}\n'.format(arch_set))
    build_model(model)
    return model


def finetuning(model, batches):
    """根据我们给的数据对模型网络进行更改

       args:
            model: keras model
            batches: keras DirectoryIterator
        returns:
            base_model: keras model, 作为 feature extractor返回
            model: keras model
    """
    n_classes = len(batches.class_indices)
    # 将最后一层输出层丢弃
    for layer in model.layers:
        layer.trainable = False
    model.layers.pop(-1)
    _input = model.input
    # 不能使用 model.output
    x = model.layers[-1].output
    x = layers.Dense(n_classes, activation='softmax')(x)
    ftmodel = models.Model(inputs=[_input], outputs=[x])
    build_model(ftmodel)
    return ftmodel


def fit_d(model, batches, valid_batches, epochs, callbacks=None):
    """模型训练"""
    model.fit_generator(batches, steps_per_epoch=batches.n // batches.batch_size,
                        validation_data=valid_batches, validation_steps=valid_batches.n // valid_batches.batch_size,
                        epochs=epochs, callbacks=callbacks)


def predict_d(model, batches):
    """模型预测"""
    y_prob = model.predict_generator(batches)
    y_pred = np.argmax(y_prob, axis=1)
    return y_prob, y_pred


def save_d(model, path='./models'):
    """模型保存"""
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'model.json'), 'wt') as f:
        f.write(model.to_json())
    model.save_weights(os.path.join(path, 'model.h5'))


def load_d(path='./models'):
    """模型加载"""
    with open(os.path.join(path, 'model.json'), 'rt') as f:
        json_string = f.read()
    model = models.model_from_json(json_string)
    model.load_weights(os.path.join(path, 'model.h5'))


