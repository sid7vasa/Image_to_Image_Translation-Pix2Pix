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
    return train_dataset, val_dataset

def visualize_datasets(train_dataset, val_dataset):
    for data in train_dataset.take(1):
        print(data[0].shape)
        print(data[1].shape)
        plt.imshow(data[1])
        plt.show()

def train(parameters, generator, discriminator, gan, dataset):
    pass


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
    train_dataset, val_dataset = get_dataset(parameters)
    print(train_dataset, val_dataset)
    visualize_datasets(train_dataset, val_dataset)
    
    
