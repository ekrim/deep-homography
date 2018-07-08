import numpy as np
import tensorflow as tf
from nvidia.dali.pipeline import Pipeline
from nvidia.dali import ops


class TFRecordPipeline(Pipeline):
  def __init__(self, batch_size, n_threads, device_id):
    super().__init__(batch_size, n_threads, device_id)
    self.input = ops.TFRecordReader(
      path=
      index_path=
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
      crop=(
                                        

