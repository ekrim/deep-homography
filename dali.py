import numpy as np
import tensorflow as tf
from nvidia.dali.pipeline import Pipeline
from nvidia.dali import ops, \
                        types, \
                        tfrecord


class TFRecordPipeline(Pipeline):
  def __init__(self, batch_size, n_threads, device_id):
    super().__init__(batch_size, n_threads, device_id)

    self.input = ops.TFRecordReader(
      path=TODO
      index_path=TODO
      features={
        'image': tfrecord.FixedLenFeature([], tfrecord.string, ''),
        'label': tfrecord.FixedLenFeature([8], tfrecord.float32, -1)})
      
      
    self.decode = ops.nvJPEGDecoder(device='mixed', output_type=types.GRAY)
    self.uniform = ops.Uniform(range=(0.0, 1.0))
    self.iter = 0

  def define_graph(self):
    inputs = self.input()
    images = self.decode(inputs['image/'])
    return (images, inputs['label/'])


if __name__ == '__main__':
  batch_size = 16
  pipe = TFRecordPipeline(batch_size=batch_size, n_threads=4, device_id=0)
  pipe.build()
  pipe_out = pipe.run()
  images, labels = pipe_out

  print(type(images))
