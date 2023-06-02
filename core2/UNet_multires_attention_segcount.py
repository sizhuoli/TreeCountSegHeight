#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 00:34:33 2021

@author: sizhuo
"""


from tensorflow.keras import models, layers
from tensorflow.keras import regularizers

import tensorflow.keras.backend as K


def UNet(input_shape,input_label_channel, layer_count=64, regularizers = regularizers.l2(0.0001), gaussian_noise=0.1, weight_file = None, inputBN = 0):
        """ Method to declare the UNet model.

        Args:
            input_shape: tuple(int, int, int, int)
                Shape of the input in the format (batch, height, width, channels).
            input_label_channel: list([int])
                list of index of label channels, used for calculating the number of channels in model output.
            layer_count: (int, optional)
                Count of kernels in first layer. Number of kernels in other layers grows with a fixed factor.
            regularizers: keras.regularizers
                regularizers to use in each layer.
            weight_file: str
                path to the weight file.
        """

        input_img1 = layers.Input((input_shape[1], input_shape[2], 5), name='Input1')
        input_img2 = layers.Input((int(input_shape[1]/2), int(input_shape[2]/2), 1), name='Input2')  # lower resolution input

        if inputBN:
            input_img1_bn = layers.BatchNormalization()(input_img1) # 256, 256, n
            input_img2_bn = layers.BatchNormalization()(input_img2)
        else:
            input_img1_bn = input_img1
            input_img2_bn = input_img2
        c11 = layers.Conv2D(1*layer_count, (3, 3), activation='relu', padding='same')(input_img1_bn)
        c11 = layers.Conv2D(1*layer_count, (3, 3), activation='relu', padding='same')(c11)
        n11 = layers.BatchNormalization()(c11) # 256, 256, 64
        p11 = layers.MaxPooling2D((2, 2))(n11) # 128, 128, 64
        c11 = models.Model(inputs=input_img1, outputs=p11)


        c12 = layers.Conv2D(1*layer_count, (3, 3), activation='relu', padding='same')(input_img2_bn)
        c12 = layers.Conv2D(1*layer_count, (3, 3), activation='relu', padding='same')(c12)
        n12 = layers.BatchNormalization()(c12) # 128, 128, 64
        c12 = models.Model(inputs=input_img2, outputs=n12)

        # concate two input sources
        c20 = layers.concatenate([c11.output, c12.output]) # 128, 128, 64+64
        #batch normal after conc?

        # on the combined features
        c2 = layers.Conv2D(2*layer_count, (3, 3), activation='relu', padding='same')(c20)
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


        u6 = attention_up_and_concate(n5, n4)
        c6 = layers.Conv2D(8*layer_count, (3, 3), activation='relu', padding='same')(u6)
        c6 = layers.Conv2D(8*layer_count, (3, 3), activation='relu', padding='same')(c6)
        #add bn here
        n6 = layers.BatchNormalization()(c6)


        u7 = attention_up_and_concate(n6, n3)
        c7 = layers.Conv2D(4*layer_count, (3, 3), activation='relu', padding='same')(u7)
        c7 = layers.Conv2D(4*layer_count, (3, 3), activation='relu', padding='same')(c7)
        n7 = layers.BatchNormalization()(c7)


        u8 = attention_up_and_concate(n7, n2)
        c8 = layers.Conv2D(2*layer_count, (3, 3), activation='relu', padding='same')(u8)
        c8 = layers.Conv2D(2*layer_count, (3, 3), activation='relu', padding='same')(c8)
        n8 = layers.BatchNormalization()(c8)


        u9 = attention_up_and_concate(n8, n11)
        c9 = layers.Conv2D(1*layer_count, (3, 3), activation='relu', padding='same')(u9)
        c9 = layers.Conv2D(1*layer_count, (3, 3), activation='relu', padding='same')(c9)
        n9 = layers.BatchNormalization()(c9)

        d = layers.Conv2D(len(input_label_channel), (1, 1), activation='sigmoid', kernel_regularizer= regularizers, name = 'output_seg')(n9)

        # density map
        d2 = layers.Conv2D(len(input_label_channel), (1, 1), activation='linear', kernel_regularizer= regularizers, name = 'output_dens')(n9)

        seg_model = models.Model(inputs=[input_img1, input_img2], outputs=[d, d2])
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
