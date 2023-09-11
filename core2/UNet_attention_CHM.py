#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 13:43:53 2021

@author: rscph
"""


from tensorflow.keras import models, layers
from tensorflow.keras import regularizers

import tensorflow.keras.backend as K


def UNet(input_shape,input_label_channel, layer_count=64, regularizers = regularizers.l2(0.0001), gaussian_noise=0.1, weight_file = None):


        input_img = layers.Input(input_shape[1:], name='Input')
        
        c11 = layers.Conv2D(1*layer_count, (3, 3), activation='relu', padding='same')(input_img)
        c11 = layers.Conv2D(1*layer_count, (3, 3), activation='relu', padding='same')(c11)
        n11 = layers.BatchNormalization()(c11) # 256, 256, 64
        p11 = layers.MaxPooling2D((2, 2))(n11) # 128, 128, 64

        # on the combined features
        c2 = layers.Conv2D(2*layer_count, (3, 3), activation='relu', padding='same')(p11)
        c2 = layers.Conv2D(2*layer_count, (3, 3), activation='relu', padding='same')(c2)
        n2 = layers.BatchNormalization()(c2)
        p2 = layers.MaxPooling2D((2, 2))(n2)

        c3 = layers.Conv2D(4*layer_count, (3, 3), activation='relu', padding='same')(p2)
        c3 = layers.Conv2D(4*layer_count, (3, 3), activation='relu', padding='same')(c3)
        n3 = layers.BatchNormalization()(c3)
        p3 = layers.MaxPooling2D((2, 2))(n3)

        c4 = layers.Conv2D(8*layer_count, (3, 3), activation='relu', padding='same')(p3)
        c4 = layers.Conv2D(8*layer_count, (3, 3), activation='relu', padding='same')(c4)
        n4 = layers.BatchNormalization()(c4)
        p4 = layers.MaxPooling2D(pool_size=(2, 2))(n4)

        c5 = layers.Conv2D(16*layer_count, (3, 3), activation='relu', padding='same')(p4)
        c5 = layers.Conv2D(16*layer_count, (3, 3), activation='relu', padding='same')(c5)
        n5 = layers.BatchNormalization()(c5)

        # add attention block here
        u6 = attention_up_and_concate(n5, n4)
        c6 = layers.Conv2D(8*layer_count, (3, 3), activation='relu', padding='same')(u6)
        c6 = layers.Conv2D(8*layer_count, (3, 3), activation='relu', padding='same')(c6)
        n6 = layers.BatchNormalization()(c6)

        u7 = attention_up_and_concate(n6, n3)
        c7 = layers.Conv2D(4*layer_count, (3, 3), activation='relu', padding='same')(u7)
        c7 = layers.Conv2D(4*layer_count, (3, 3), activation='relu', padding='same')(c7)
        n7 = layers.BatchNormalization()(c7)

        u8 = attention_up_and_concate(n7, n2)
        c8 = layers.Conv2D(2*layer_count, (3, 3), activation='relu', padding='same')(u8)
        c8 = layers.Conv2D(2*layer_count, (3, 3), activation='relu', padding='same')(c8)
        n8 = layers.BatchNormalization()(c8)
        
        # relu activation or linear
        d = layers.Conv2D(len(input_label_channel), (1, 1), activation='linear', kernel_regularizer= regularizers)(n8)

        seg_model = models.Model(inputs=[input_img], outputs=[d])
        if weight_file:
            seg_model.load_weights(weight_file)
        seg_model.summary()
        return seg_model


def attention_up_and_concate(down_layer, layer):
    
    in_channel = down_layer.get_shape().as_list()[3]
    up = layers.UpSampling2D(size=(2, 2))(down_layer)
    layer = attention_block_2d(x=layer, g=up, inter_channel=in_channel // 4)
    my_concat = layers.Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))
    concate = my_concat([up, layer])
    return concate


def attention_block_2d(x, g, inter_channel):
    theta_x = layers.Conv2D(inter_channel, [1, 1], strides=[1, 1])(x)
    phi_g = layers.Conv2D(inter_channel, [1, 1], strides=[1, 1])(g)
    f = layers.Activation('relu')(layers.add([theta_x, phi_g]))
    psi_f = layers.Conv2D(1, [1, 1], strides=[1, 1])(f)
    rate = layers.Activation('sigmoid')(psi_f)
    att_x = layers.multiply([x, rate])

    return att_x