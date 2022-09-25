# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 22:44:57 2022

@author: santosh
"""
from data.dataset import generate_tfrecords, load_tfrecords
from tensorflow.keras.utils import plot_model
from models.pi2pix import Discriminator, Generator, GAN
import yaml
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('WARNING')


print(tf.test.is_gpu_available())


def get_dataset(parameters):
    if not os.path.exists(os.path.join(
            parameters['dataset']['data_dir']['train'],
            'train.tfrecords')) or not os.path.exists(os.path.join(
                parameters['dataset']['data_dir']['val'], 'val.tfrecords')):
        print("Generating TF Records:")
        generate_tfrecords(parameters)
    else:
        print("Using existing TF Records")
    train_dataset, val_dataset = load_tfrecords(parameters)
    train_dataset = train_dataset.batch(parameters['dataset']['batch_size'])
    train_dataset = train_dataset.shuffle(buffer_size=100)
    val_dataset = val_dataset.batch(parameters['dataset']['batch_size'])
    return train_dataset, val_dataset


def visualize_datasets(train_dataset, val_dataset):
    for data in train_dataset.take(1):
        print(data[0].shape)
        print(data[1].shape)
        picture = (data[1]*127.5) + 127.5
        picture = np.array(picture, dtype=np.uint8)
        plt.imshow(picture)
        plt.show()


def train(parameters, generator, discriminator, gan, train_dataset, val_dataset, epochs=10, batch_size=1):
    

    pass


if __name__ == "__main__":
    # Parameters for the training sesssion:
    with open('parameters.yaml', 'r') as file:
        parameters = yaml.safe_load(file)

    # Creating/Loading TF Records data
    train_dataset, val_dataset = get_dataset(parameters)

    # Getting the models - Generator, Discriminator, GAN(Combined)
    generator = Generator((256, 256, 3)).get_model()
    discriminator = Discriminator((256, 256, 3), (256, 256, 3)).get_model()
    gan = GAN(generator, discriminator, (256, 256, 3)).get_model()

    # Visualize or not:
    if parameters['visualize']:
        # Visualize data after storing and loading
        visualize_datasets(train_dataset, val_dataset)
        print(discriminator.summary())
        plot_model(discriminator, to_file="discriminator.png")
        print(generator.summary())
        plot_model(generator, to_file="generator.png")
        print(gan.summary())
        plot_model(gan, to_file="gan.png")

    train(parameters, generator, discriminator, gan, train_dataset)
