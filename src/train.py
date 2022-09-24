# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 22:44:57 2022

@author: santosh
"""
from tensorflow.keras.utils import plot_model
from models.pi2pix import Discriminator, Generator, GAN
import yaml
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('WARNING')


print(tf.test.is_gpu_available())


def run(parameters):
    disc = Discriminator((256, 256, 3)).get_model()
    return disc


if __name__ == "__main__":
    with open('parameters.yaml', 'r') as file:
        parameters = yaml.safe_load(file)
    discriminator = Discriminator((256, 256, 3), (256, 256, 3)).get_model()
    print(discriminator.summary())
    plot_model(discriminator, to_file="discriminator.png")
    generator = Generator((256, 256, 3)).get_model()
    print(generator.summary())
    plot_model(generator, to_file="generator.png")
    gan = GAN(generator, discriminator, (256, 256, 3)).get_model()
    print(gan.summary())
    plot_model(gan, to_file="gan.png")
