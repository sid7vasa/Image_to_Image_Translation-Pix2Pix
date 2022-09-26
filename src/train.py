# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 22:44:57 2022

@author: Santosh Vasa
Name: Santosh Vasa<br>
Date: September 25, 2022<br>
Class: Advanced Perception
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
import traceback


print(tf.test.is_gpu_available())


def get_dataset(parameters):
    """
    Loads TF records (If not exists, creates TF Records).
    Preprocesses datset to the needed format.
    returns train and validation dataset tf.data instances.

    Parameters
    ----------
    parameters : parameters loaded from the parameters.yaml file.
        DESCRIPTION.

    Returns
    -------
    train_dataset : tf.data train dataset instance
        DESCRIPTION.
    val_dataset : tf.data validation dataset instance
        DESCRIPTION.

    """
    if not os.path.exists(os.path.join(
            parameters['dataset']['data_dir']['train'],
            'train.tfrecords')) or not os.path.exists(os.path.join(
                parameters['dataset']['data_dir']['val'], 'val.tfrecords')):
        print("Generating TF Records:")
        generate_tfrecords(parameters)
    else:
        print("Using existing TF Records")
    train_dataset, val_dataset = load_tfrecords(parameters)
    train_dataset = train_dataset.batch(
        parameters['dataset']['batch_size']).shuffle(buffer_size=100)
    val_dataset = val_dataset.batch(parameters['dataset']['batch_size'])
    return train_dataset, val_dataset


def visualize_datasets(train_dataset, val_dataset):
    """
    Visualize an example in the dataset by reversing the preprocessing steps.
    using matplotlib.

    Parameters
    ----------
    train_dataset : tf.data training instance
        DESCRIPTION.
    val_dataset : tf.data validation instance
        DESCRIPTION.

    Returns
    -------
    None.

    """
    for data in train_dataset.take(1):
        print(data[0].shape)
        print(data[1].shape)
        picture = data[1].numpy()[0]
        picture = (picture*127.5) + 127.5
        picture = np.array(picture, dtype=np.uint8)
        plt.imshow(picture)
        plt.show()

def plot_sample_outputs(dataset, val=False):
    """
    Takes random examples from the input tf.data instance and then plots the 
    generated output, corresponding inputs and ground truths.

    Parameters
    ----------
    dataset : tf.data validation instance
        DESCRIPTION.
    val : is validation dataset instance

    Returns
    -------
    img : TYPE
        DESCRIPTION.

    """
    _, axs = plt.subplots(1, 3, figsize=(8, 24))
    axs = axs.flatten()
    def un_normalize(img):
        img = (img * 127.5) + 127.5
        img = np.array(img, dtype=np.uint8)[0]
        return img
    if val:
      dataset = dataset.shuffle(buffer_size=100)

    for data in dataset.take(1):
        x_fake = un_normalize(generator(data[0]))
        x_real_a = un_normalize(data[0])
        x_real_b= un_normalize(data[1])
        imgs = [x_real_a, x_real_b, x_fake]
        for ax, img in zip(axs, imgs):
            ax.imshow(img)
        plt.show()

def train(parameters, generator, discriminator, gan, train_dataset, val_dataset, epochs=100):
    """
    Training loop for the GAN model. Doesn't use the fit function. 

    Parameters
    ----------
    parameters : parameters loaded form json - parameters.yaml
        DESCRIPTION.
    generator : generator instance
        DESCRIPTION.
    discriminator : discriminator instance 
        DESCRIPTION.
    gan : TYPE
        DESCRIPTION.
    train_dataset : tf.data train dataset instance
        DESCRIPTION.
    val_dataset : tf.data validation dataset instance
        DESCRIPTION.
    epochs : number of epochs to repeat the training, optional
        DESCRIPTION. The default is 100.

    Returns
    -------
    None.

    """
    n_patch = discriminator.output_shape[1]
    train_dataset = train_dataset.repeat(epochs)
    batch_size = parameters['dataset']['batch_size']
    for step, input_output_data in enumerate(train_dataset):
        
        # Preparing the inputs and outputs for the GAN model.
        y_real = np.ones((batch_size, n_patch, n_patch, 1))
        y_fake = np.zeros((batch_size, n_patch, n_patch, 1))

        # Uncomment to add noise to the labels for discriminator. 
        # y_real += 0.05 * tf.random.uniform(y_real.shape)
        # y_fake += 0.05 * tf.random.uniform(y_fake.shape)

        x_real_a = input_output_data[0]
        x_real_b = input_output_data[1]
        x_fake_b = generator(x_real_a)
        
        # Sometimes, if the last batch len doesn't equate to the batch_size, throws an error.
        # Todo: Handle it better.
        try:
          for layer in discriminator.layers:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
              layer.trainable = True

          d_loss1 = discriminator.train_on_batch([x_real_a, x_real_b], y_real)
          d_loss2 = discriminator.train_on_batch([x_real_a, x_fake_b], y_fake)
          for layer in discriminator.layers:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
              layer.trainable = False

          g_loss, _, _ = gan.train_on_batch(x_real_a, [x_real_b, y_real])
        except:
          traceback.print_exc()

        # Saves model every 500 iterations and logs information every 100 iterations
        # Todo: Can be handled better.
        if step % 100 == 0:
            print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (step+1, d_loss1, d_loss2, g_loss))
            plot_sample_outputs(train_dataset)
            plot_sample_outputs(val_dataset, val=True)
        if step % 501 == 0:
            print("Saving models:")
            generator.save("generator.h5")
            discriminator.save("discriminator.h5")
            gan.save("gan.h5")


if __name__ == "__main__":
    # Parameters for the training sesssion:
    with open('../parameters.yaml', 'r') as file:
        parameters = yaml.safe_load(file)

    # Creating/Loading TF Records data
    train_dataset, val_dataset = get_dataset(parameters)

    # Getting the models - Generator, Discriminator, GAN(Combined)
    generator = Generator((256, 256, 3)).get_model()
    discriminator = Discriminator((256, 256, 3), (256, 256, 3)).get_model()
    gan = GAN(generator, discriminator, (256, 256, 3)).get_model()

    # Visualize or not:
    # Plots the model architecture on a png image. 
    if parameters['visualize']:
        # Visualize data after storing and loading
        visualize_datasets(train_dataset, val_dataset)
        print(discriminator.summary())
        plot_model(discriminator, to_file="discriminator.png")
        print(generator.summary())
        plot_model(generator, to_file="generator.png")
        print(gan.summary())
        plot_model(gan, to_file="gan.png")

    train(parameters, generator, discriminator,
          gan, train_dataset, val_dataset)
