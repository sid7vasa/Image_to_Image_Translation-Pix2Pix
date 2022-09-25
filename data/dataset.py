# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 22:46:17 2022

@author: santo
"""
import tensorflow as tf
import numpy as np
from imutils import paths
import os
from PIL import Image


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):  # if value ist tensor
        value = value.numpy()  # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array


def preprocess_data(X1, X2):
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]


def parse_single_image(image, label):
    data = {
        'height': _int64_feature(image.shape[0]),
        'width': _int64_feature(image.shape[1]),
        'depth': _int64_feature(image.shape[2]),
        'raw_image': _bytes_feature(serialize_array(image)),
        'l_height': _int64_feature(label.shape[0]),
        'l_width': _int64_feature(label.shape[1]),
        'l_depth': _int64_feature(label.shape[2]),
        'l_raw_image': _bytes_feature(serialize_array(label)),
    }

    out = tf.train.Example(features=tf.train.Features(feature=data))
    return out


def parse_tfr_element(element):
    # use the same structure as above; it's kinda an outline of the structure we now want to create
    data = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'raw_image': tf.io.FixedLenFeature([], tf.string),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'l_height': tf.io.FixedLenFeature([], tf.int64),
        'l_width': tf.io.FixedLenFeature([], tf.int64),
        'l_raw_image': tf.io.FixedLenFeature([], tf.string),
        'l_depth': tf.io.FixedLenFeature([], tf.int64),
    }

    content = tf.io.parse_single_example(element, data)

    height = content['height']
    width = content['width']
    depth = content['depth']
    raw_image = content['raw_image']
    l_height = content['l_height']
    l_width = content['l_width']
    l_depth = content['l_depth']
    l_raw_image = content['l_raw_image']

    # get our 'feature'-- our image -- and reshape it appropriately
    feature = tf.io.parse_tensor(raw_image, out_type=tf.uint8)
    feature = tf.reshape(feature, shape=[height, width, depth])
    label = tf.io.parse_tensor(l_raw_image, out_type=tf.uint8)
    label = tf.reshape(label, shape=[l_height, l_width, l_depth])
    return (feature, label)


def split_image(image_path):
    image = Image.open(image_path)
    image = image.resize((512, 256))
    image = np.asarray(image)
    image, label = image[:, :256], image[:, 256:]
    return image, label


def write_images_to_tfr_short(store_dir, image_paths, filename: str = "train"):
    """ 
    Modify this function to change the dataset input format.
    """
    filename = filename+".tfrecords"
    writer = tf.io.TFRecordWriter(os.path.join(
        store_dir, filename))
    count = 0

    for image_path in image_paths:
        current_image, current_label = split_image(image_path)
        out = parse_single_image(image=current_image, label=current_label)
        writer.write(out.SerializeToString())
        count += 1

    writer.close()
    print(f"Wrote {count} elements to TFRecord")
    return count


def generate_tfrecords(parameters):
    train_paths = paths.list_images(parameters['dataset']['data_dir']['train'])
    val_paths = paths.list_images(parameters['dataset']['data_dir']['val'])
    count_train = write_images_to_tfr_short(
        parameters['dataset']['data_dir']['train'], train_paths, filename="train")
    count_val = write_images_to_tfr_short(
        parameters['dataset']['data_dir']['val'], val_paths, filename='val')
    print("Number of images in Train:", count_train)
    print("Number of images in Val:", count_val)
    return


def load_tfrecords(parameters):
    train_dataset = tf.data.TFRecordDataset(os.path.join(
        parameters['dataset']['data_dir']['train'], 'train.tfrecords'))
    val_dataset = tf.data.TFRecordDataset(os.path.join(
        parameters['dataset']['data_dir']['val'], 'val.tfrecords'))

    train_dataset = train_dataset.map(parse_tfr_element)
    train_dataset = train_dataset.map(lambda x, y: (tf.cast(x, tf.float32), (tf.cast(y, tf.float32))))
    train_dataset = train_dataset.map(preprocess_data)

    val_dataset = val_dataset.map(parse_tfr_element)
    val_dataset = val_dataset.map(lambda x, y: (tf.cast(x, tf.float32), (tf.cast(y, tf.float32))))
    val_dataset = val_dataset.map(preprocess_data)

    return (train_dataset, val_dataset)
