import numpy as np
import tensorflow as tf
from nvidia.dali.pipeline import Pipeline
from nvidia.dali import ops


class TFRecordPipeline(Pipeline):
  def __init__(self, batch_size, n_threads, device_id):
    super().__init__(batch_size, n_threads, device_id)

    self.input = ops.TFRecordReader(
      path=TODO
      index_path=TODO
      features={})

    self.decode = ops.nvJPEGDecoder(device='mixed', output_type=types.RGB)

    self.resize = ops.Resize(
      device='gpu',
      random_resize=True,
      resize_a=32,
      resize_b=32,
      image_type=types.RGB,
      interp_type=types.INTERP_LINEAR)

    self.cmn = ops.CropMirrorNormalize(
      device='gpu',
      output_dtype=types.FLOAT,
      crop=(),
      image_type=types.RGB,
      mean=[128., 128., 128.]
      std=[1., 1., 1.])

    self.uniform = ops.Uniform(range=(0.0, 1.0))

  def define_graph(self):
    inputs = self.input()
    images = self.decode(inputs['image/'])
    resized_images = self.resize(images)
    output = self.cmn(
      resized_images, 
      crop_pos_x=self.uniform(),
      crop_pos_y=self.uniform())
    return (output, inputs['image/class/text'].gpu())


class CelebAInput:

  def __init__(self, crop_size=64):
    self.HEIGHT_ORIG = 218
    self.WIDTH_ORIG = 178

    self.HEIGHT = crop_size
    self.WIDTH = crop_size
    self.DEPTH = 3
    
    self.read_buffer = 3*64*64*2**13
    self.shuffle_buffer = 2**15
    self.file_list = ['data/img_align_celeba.tfrecords']

  def input_fn(self, mode='test', epochs=1, batch_size=64):
    file_list = self.file_list
    if type(file_list) is list and len(file_list)>1:
      print('\n***processing in multiple file interleave mode***\n')
      file_list = self.file_list
      dataset = tf.data.Dataset.list_files(file_list, shuffle=True)
    
      if mode == 'train':
        dataset = dataset.shuffle(
	  buffer_size=min(len(file_list), self.shuffle_buffer)).repeat()
    
      def process_tfrecord(file_name):
        dataset = tf.data.TFRecordDataset(
	  file_name, 
	  buffer_size=self.read_buffer)
        return dataset

      dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
          process_tfrecord, cycle_length=4, sloppy=True))
    else:
      print('\n*** processing in single file mode ***\n')
      dataset = tf.data.TFRecordDataset(
        file_list, 
	buffer_size=self.read_buffer)
    
    if mode=='train':
      dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(
        self.shuffle_buffer, 
	count=epochs))

    dataset = dataset.map(
      self.make_parser(mode), 
      num_parallel_calls=8)

    dataset = dataset.batch(batch_size)
    #dataset = dataset.apply(
    #  tf.contrib.data.batch_and_drop_remainder(batch_size))

    dataset = dataset.prefetch(batch_size)
   
    iterator = dataset.make_initializable_iterator()
    image = iterator.get_next()
    
    return image, iterator
    
  def make_parser(self, mode):
    def parser_fn(value):
      keys_to_features = {
        'image': tf.FixedLenFeature((), tf.string, '')}
      parsed = tf.parse_single_example(value, features=keys_to_features)
      image = tf.decode_raw(parsed['image'], tf.uint8)
      image.set_shape([self.HEIGHT_ORIG * self.WIDTH_ORIG * self.DEPTH])
      
      image = tf.cast(image, tf.float32)
     
      image = tf.transpose(
        tf.reshape(2*image/255.0 - 1, (self.DEPTH, self.HEIGHT_ORIG, self.WIDTH_ORIG)), 
        (1,2,0))
      
      image = tf.image.resize_images(
                image, 
                (self.HEIGHT, self.WIDTH), 
                method=tf.image.ResizeMethod.BICUBIC)

      return image
    return parser_fn
