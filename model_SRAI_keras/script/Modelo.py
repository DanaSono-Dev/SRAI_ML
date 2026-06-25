#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 11:15:12 2026

@author: multi4090
"""

import os
from dotenv import load_dotenv

load_dotenv()

# from tensorflow.keras.optimizers.schedules import ExponentialDecay, CosineDecay
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts

from keras.optimizers import Adam
from keras.layers import Dense, Flatten, Conv2D
from keras.layers import Dropout, ReLU, LeakyReLU, Input
from keras.layers import MaxPooling2D, BatchNormalization

from keras.models import Model, Sequential

# from tensorflow.keras.regularizers import l2
from keras import regularizers

# Configuración desde variables de entorno
INITIAL_LR   = float(os.getenv('INITIAL_LR',   2e-4))
COSINE_T_MUL = float(os.getenv('COSINE_T_MUL', 1.5))
COSINE_M_MUL = float(os.getenv('COSINE_M_MUL', 1.5))
COSINE_ALPHA = float(os.getenv('COSINE_ALPHA',  1e-9))
ADAM_BETA1   = float(os.getenv('ADAM_BETA1',    0.5))
ADAM_BETA2   = float(os.getenv('ADAM_BETA2',    0.8))
DROPOUT_RATE = float(os.getenv('DROPOUT_RATE',  0.3))
NUM_CLASSES  = int(os.getenv('NUM_CLASSES',      4))

def opti(fdecay):
    #lr_schedule = ExponentialDecay(initial_learning_rate=0.0005,
    #                                decay_steps=220,
    #                                decay_rate=0.92,
    #                                staircase=True)

    lr_schedule = CosineDecayRestarts(initial_learning_rate=INITIAL_LR,
                                      first_decay_steps=fdecay, t_mul=COSINE_T_MUL,
                                      m_mul=COSINE_M_MUL, alpha=COSINE_ALPHA,
                                      )

    opt = Adam(learning_rate=lr_schedule,
               beta_1=ADAM_BETA1, beta_2=ADAM_BETA2,
               # weight_decay=4e-04,
               # amsgrad=True,
               # ema_momentum=0.999,
               # use_ema=True,
              )

    return opt

def Clasificador(shape, fdecay):    # 356, 637, 3
    in_shape=shape
    inputs = Input(shape=in_shape, name='InDis')
    '''
    # aumen = Sequential([
    #     #layers.RandomFlip("horizontal"),
    #     #tf.keras.layers.RandomRotation(0.1), # Rota +/- 10%
    #     #tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    #     # tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1),
    #     # tf.keras.layers.RandomContrast(factor=0.1, value_range=(0, 1)),
    #     # tf.keras.layers.RandomBrightness(0.4, value_range=(0, 1)), # Puedes añadirla también
    #     tf.keras.layers.Equalization(value_range=(0, 255))
    #     ], name="data_augmentation")

    # inputs = aumen(inputs)
    '''
    do = DROPOUT_RATE

    # Calculo de numero de parametros
    # (SizeK x SizeK) x NumK x NumCapPrev + NumK
    # Num de parametros = 31,232
    x = Conv2D(128, (7, 7), padding='same')(inputs)
    x = MaxPooling2D(pool_size=(2,2), padding='same', strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(do)(x)
    print('Primer reduccion: ', x.shape)
    #

    # Num de parametros = 401,472
    x = Conv2D(64, (7, 7), padding='same')(x)
    x = MaxPooling2D(pool_size=(2,2), padding='same', strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(do)(x)


    # Num de parametros = 51,232
    x = Conv2D(32, (5, 5), padding='same')(x)
    x = MaxPooling2D(pool_size=(2,2), padding='same', strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(do)(x)

    # Num de parametros = 51,232
    x = Conv2D(10, (5, 5), padding='same')(x)
    x = MaxPooling2D(pool_size=(2,2), padding='same', strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(do)(x)

    # # Num de parametros =4,624
    # x = Conv2D(12, (3, 3),  padding='same')(x)
    # x = MaxPooling2D(pool_size=(2,2), padding='same', strides=(2, 2))(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=0.2)(x)
    # x = Dropout(do)(x)
    print('Ultimo reduccion: ', x.shape)

    x = Flatten(name = 'Flatten')(x)
    print('Flatten         : ', x.shape)

    x = Dense(x.shape[1]//2, activation='relu',
                   # kernel_regularizer=regularizers.l2(0.001),
                   # bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.001),
                  name = 'Dense_0')(x)
    x = BatchNormalization()(x)
    x = Dropout(do)(x)

    # x = Dense(x.shape[1]//2, activation='selu',
    #               # kernel_regularizer=regularizers.l2(0.001),
    #               # bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.001),
    #               name = 'Dense_1')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(do)(x)

    x = Dense(x.shape[1]//4, activation='relu',
                  # kernel_regularizer=regularizers.l2(0.001),
                  # bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.001),
                  name = 'Dense_2')(x)
    x = BatchNormalization()(x)
    x = Dropout(do)(x)

    x = Dense(NUM_CLASSES, activation='softmax',
                  #kernel_regularizer=regularizers.l2(0.001),
                  # bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.001),
                  name='DenseOutput')(x)

    model = Model(inputs=inputs, outputs=x)

    # compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer=opti(fdecay),
                  metrics=['categorical_accuracy'])

    return model

if __name__ == "__main__":

    shape = (224, 224, 3)
    Images  = int(os.getenv('TEST_IMAGES_COUNT',  1000))
    batch0  = int(os.getenv('TEST_BATCH_SIZE',    16))
    Efdecay = int(os.getenv('TEST_EPOCH_FDECAY',  10))
    fdecay = Efdecay * (Images // batch0)
    model = Clasificador(shape, fdecay)
    print(model.summary())

    print(fdecay)
