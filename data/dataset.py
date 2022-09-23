# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 22:46:17 2022

@author: santo
"""
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from imutils import paths
import yaml


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array

def parse_single_image(image, label):
  data = {
        'height' : _int64_feature(image.shape[0]),
        'width' : _int64_feature(image.shape[1]),
        'depth' : _int64_feature(image.shape[2]),
        'raw_image' : _bytes_feature(serialize_array(image)),
        # 'label' : _int64_feature(label),
        'l_height' : _int64_feature(label.shape[0]),
        'l_width' : _int64_feature(label.shape[1]),
        'l_depth' : _int64_feature(label.shape[2]),
        'l_raw_image' : _bytes_feature(serialize_array(label)),
    }

  out = tf.train.Example(features=tf.train.Features(feature=data))
  return out

def write_images_to_tfr_short(images, labels, filename:str="train"):
  filename= filename+".tfrecords"
  writer = tf.io.TFRecordWriter(filename) 
  count = 0

  for index in range(len(images)):
    current_image = images[index] 
    current_label = labels[index]

    out = parse_single_image(image=current_image, label=current_label)
    writer.write(out.SerializeToString())
    count += 1

  writer.close()
  print(f"Wrote {count} elements to TFRecord")
  return count


def generate_tfrecords(parameters):
    image_paths = paths.list_images(parameters['dataset']['data_dir']['train'])
    
    count = write_images_to_tfr_short(images_small, labels_small, filename="train")
    return
