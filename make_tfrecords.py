import struct
import os
import urllib
import zipfile
import pickle
import PIL
import numpy as np
import tensorflow as tf


def jpg_reader(filename):
  img = PIL.Image.open(filename)
  return np.asarray(img)


def to_tfrecords():
  img_dir = 'data/synth_data'
  
  assert os.path.isdir(img_dir), 'Download the MSCOCO dataset and prepare it' 
  
  with open('data/label_file.txt', 'r') as f:
    label_list = [line.rstrip().rstrip(',').split(';') for line in f]
   
  output_file = 'data/homography.tfrecords'

  with tf.python_io.TFRecordWriter(output_file) as record_writer:
    for num, label_str in label_list:
      label = np.array([float(el) for el in label_str.split(',')]).astype(np.float32)
      input_file_orig = 'data/synth_data/{:s}_orig.jpg'.format(num)
      input_file_warp = 'data/synth_data/{:s}_warp.jpg'.format(num)
      img_orig = jpg_reader(input_file_orig)
      img_warp = jpg_reader(input_file_warp)
      img = np.concatenate([img_orig[:,:,None], img_warp[:,:,None]], axis=2)

      example = tf.train.Example(
        features=tf.train.Features(
          feature={
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()])),
            'label': tf.train.Feature(float_list=tf.train.FloatList(value=label))
        }))

      record_writer.write(example.SerializeToString())


if __name__ == '__main__':
  to_tfrecords()
