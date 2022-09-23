# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 22:44:57 2022

@author: santo
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('WARNING')
import numpy
import yaml
from models.pi2pix import Discriminator, Generator
from tensorflow.keras.utils import plot_model


print(tf.test.is_gpu_available())


def run(parameters):
    disc = Discriminator((256, 256, 3)).get_model()
    return disc


if __name__ == "__main__":
    with open('../parameters.yaml', 'r') as file:
        parameters = yaml.safe_load(file)
    disc = run(parameters)
    print(disc.summary())
    generator = Generator((256,256,3)).get_model()
    print(generator.summary())
    plot_model(generator, to_file="../generator.png")
