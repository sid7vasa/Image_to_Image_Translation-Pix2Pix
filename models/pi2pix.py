# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 22:45:54 2022

@author: Santosh Vasa
Name: Santosh Vasa<br>
Date: September 25, 2022<br>
Class: Advanced Perception
"""
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization


def ConvBNRelu(filters=64, kernel_size=(4, 4), stride=(2, 2), padding="same", init=RandomNormal(stddev=0.2), batch_norm=True):
    """
    A custom layer for Convolution, Batch Norm and ReLu activation. 
    This combination repeats multiple times in the model definition.
    Parameters 
    ----------
    filters : Convolutional layer number of filters
        DESCRIPTION. The default is 64.
    kernel_size : Convolutional layer kernel size
        DESCRIPTION. The default is (4, 4).
    stride : Convolutional layer stride
        DESCRIPTION. The default is (2, 2).
    padding : Convolutional layer padding
        DESCRIPTION. The default is "same".
    init : parameter initializer
        DESCRIPTION. The default is RandomNormal(stddev=0.2).
    batch_norm : should it have a batch norm layer
        DESCRIPTION. The default is True.

    Returns
    -------
    block : Activation output
    """
    block = tf.keras.Sequential()
    block.add(Conv2D(filters, kernel_size, strides=stride,
              padding=padding, kernel_initializer=init))
    if batch_norm:
        block.add(BatchNormalization())
    block.add(LeakyReLU(alpha=0.2))
    return block


def decoder_block(inputs, skip_inputs, filters, dropout=True):
    """
    Decoder block for the decoder in the U-Net of the generator.

    Parameters
    ----------
    inputs : Input from the previous layer.
        DESCRIPTION. Functional input from previous layer.
    skip_inputs : Residual skip connections from encoder blocks. 
        DESCRIPTION.
    filters : number of filters convolutional layer
        DESCRIPTION.
    dropout : dropout boolean to have it or not.
        DESCRIPTION. The default is True.

    Returns
    -------
    g : activation output after forward pass of the block.
        DESCRIPTION.

    """
    init = RandomNormal(stddev=0.02)
    g = Conv2DTranspose(filters, (4, 4), strides=(
        2, 2), padding='same', kernel_initializer=init)(inputs)
    if dropout:
        g = Dropout(0.5)(g, training=True)
    g = Concatenate()([g, skip_inputs])
    g = Activation('relu')(g)
    return g


class Generator():
    def __init__(self, input_shape):
        """
        Generator constructor.

        Parameters
        ----------
        input_shape : requires input shape for static graph building.
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.input_shape = input_shape
        self.init = RandomNormal(stddev=0.2)

    def get_model(self):
        """
        Constructs the static graph and returns the generator model object.

        Returns
        -------
        model : Generator object
            DESCRIPTION.

        """
        inputs = Input(shape=self.input_shape)
        e1 = ConvBNRelu(filters=64, batch_norm=False)(inputs)
        e2 = ConvBNRelu(filters=128)(e1)
        e3 = ConvBNRelu(filters=256)(e2)
        e4 = ConvBNRelu(filters=512)(e3)
        e5 = ConvBNRelu(filters=512)(e4)
        e6 = ConvBNRelu(filters=512)(e5)
        e7 = ConvBNRelu(filters=512)(e6)
        bottle_neck = Conv2D(512, (4, 4), strides=(
            2, 2), padding='same', kernel_initializer=self.init)(e7)
        a = Activation('relu')(bottle_neck)
        d1 = decoder_block(a, e7, 512)
        d2 = decoder_block(d1, e6, 512)
        d3 = decoder_block(d2, e5, 512)
        d4 = decoder_block(d3, e4, 512, dropout=False)
        d5 = decoder_block(d4, e3, 256, dropout=False)
        d6 = decoder_block(d5, e2, 128, dropout=False)
        d7 = decoder_block(d6, e1, 64, dropout=False)
        conv = Conv2DTranspose(self.input_shape[2], (4, 4), strides=(
            2, 2), padding='same', kernel_initializer=self.init)(d7)
        out = Activation('tanh')(conv)
        model = tf.keras.Model(inputs, out)
        return model


class Discriminator():
    def __init__(self, input_shape, gen_shape):
        """
        Constructor for the discriminator.

        Parameters
        ----------
        input_shape : model inputs shape for static graph building.
            DESCRIPTION.
        gen_shape : generator output shape/ label shape.
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.input_shape = input_shape
        self.gen_shape = gen_shape
        self.init = RandomNormal(stddev=0.2)

    def get_model(self):
        """
        Constructs the static graph and returns the discriminator model object.

        Returns
        -------
        model : discriminator
            DESCRIPTION.

        """
        inputs = Input(shape=self.input_shape)
        gen_ins = Input(shape=self.gen_shape)
        x = Concatenate()([inputs, gen_ins])
        x = ConvBNRelu(filters=64, batch_norm=False)(x)
        x = ConvBNRelu(filters=128)(x)
        x = ConvBNRelu(filters=256)(x)
        x = ConvBNRelu(filters=512)(x)
        x = ConvBNRelu(filters=512)(x)
        x = Conv2D(1, (4, 4), padding='same',
                   kernel_initializer=self.init)(x)
        out = Activation('sigmoid')(x)
        model = tf.keras.Model([inputs, gen_ins], out)
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy',
                      optimizer=opt, loss_weights=[0.5])
        return model


class GAN():
    def __init__(self, generator, discriminator, input_shape):
        """
        Comnined model of discriminator and generator. Used for generator training.

        Parameters
        ----------
        generator : generator instance
            DESCRIPTION.
        discriminator : discriminator instance
            DESCRIPTION.
        input_shape : input data shape
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.generator = generator
        self.discriminator = discriminator
        self.input_shape = input_shape

    def get_model(self):
        """
        Constructs the static graph and returns the combined model object.

        Returns
        -------
        model : combined model instance.
            DESCRIPTION.

        """
        inputs = Input(shape=self.input_shape)
        gen_out = self.generator(inputs)
        disc_out = self.discriminator([inputs, gen_out])
        model = tf.keras.Model(inputs, [gen_out, disc_out])
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss=['mae', 'binary_crossentropy'],
                      optimizer=opt, loss_weights=[100, 1])
        return model
