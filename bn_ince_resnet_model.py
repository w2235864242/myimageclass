# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use input() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile

import tensorflow.python.platform
from six.moves import urllib
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

slim = tf.contrib.slim

#from tensorflow.models.image.cifar10 import cifar10_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """Number of images to process in a batch.""")
# tf.app.flags.DEFINE_string('data_dir', 'cifar10_data/',
#                            """Path to the CIFAR-10 data directory.""")


# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 299

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 3
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1101
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 219

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = IMAGE_SIZE
NUM_CLASSES = NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 35.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01       # Initial learning rate.

# If a model is trained with multiple GPU's prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))



def inception_resnet_v2_arg_scope(weight_decay=0.00004,
                        use_batch_norm=True,
                        batch_norm_decay=0.9997,
                        batch_norm_epsilon=0.001,
                        is_training=True):
  """Defines the default arg scope for inception models.

  Args:
    weight_decay: The weight decay to use for regularizing the model.
    use_batch_norm: "If `True`, batch_norm is applied after each convolution.
    batch_norm_decay: Decay for batch norm moving average.
    batch_norm_epsilon: Small float added to variance to avoid dividing by zero
      in batch norm.

  Returns:
    An `arg_scope` to use for the inception models.
  """
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': batch_norm_decay,
      # epsilon to prevent 0s in variance.
      'epsilon': batch_norm_epsilon,
      # collection containing update_ops.
      'updates_collections': tf.GraphKeys.UPDATE_OPS,

      'is_training': is_training
  }
  if use_batch_norm:
    # normalizer_fn = tf.contrib.layers.batch_norm
    normalizer_fn = slim.batch_norm
    normalizer_params = batch_norm_params
  else:
    normalizer_fn = None
    normalizer_params = {}
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_regularizer=slim.l2_regularizer(weight_decay)):
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=slim.variance_scaling_initializer(),
        activation_fn=tf.nn.relu,
        normalizer_fn=normalizer_fn,
        normalizer_params=normalizer_params) as sc:
      return sc

#
# def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
#   """Builds the 35x35 resnet block."""
#   with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
#     with tf.variable_scope('Branch_0'):
#       tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
#     with tf.variable_scope('Branch_1'):
#       tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
#       tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
#     with tf.variable_scope('Branch_2'):
#       tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
#       tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, 3, scope='Conv2d_0b_3x3')
#       tower_conv2_2 = slim.conv2d(tower_conv2_1, 64, 3, scope='Conv2d_0c_3x3')
#     mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_1, tower_conv2_2])
#     up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
#                      activation_fn=None, scope='Conv2d_1x1')
#     net += scale * up
#     if activation_fn:
#       net = activation_fn(net)
#   return net
#
#
# def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
#   """Builds the 17x17 resnet block."""
#   with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
#     with tf.variable_scope('Branch_0'):
#       tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
#     with tf.variable_scope('Branch_1'):
#       tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
#       tower_conv1_1 = slim.conv2d(tower_conv1_0, 160, [1, 7],
#                                   scope='Conv2d_0b_1x7')
#       tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [7, 1],
#                                   scope='Conv2d_0c_7x1')
#     mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
#     up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
#                      activation_fn=None, scope='Conv2d_1x1')
#     net += scale * up
#     if activation_fn:
#       net = activation_fn(net)
#   return net
#
#
# def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
#   """Builds the 8x8 resnet block."""
#   with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
#     with tf.variable_scope('Branch_0'):
#       tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
#     with tf.variable_scope('Branch_1'):
#       tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
#       tower_conv1_1 = slim.conv2d(tower_conv1_0, 224, [1, 3],
#                                   scope='Conv2d_0b_1x3')
#       tower_conv1_2 = slim.conv2d(tower_conv1_1, 256, [3, 1],
#                                   scope='Conv2d_0c_3x1')
#     mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
#     up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
#                      activation_fn=None, scope='Conv2d_1x1')
#     net += scale * up
#     if activation_fn:
#       net = activation_fn(net)
#   return net
#
#
# def inference(inputs, num_classes=1001, is_training=True,
#                         dropout_keep_prob=0.8,
#                         reuse=None,
#                         scope='InceptionResnetV2'):
#   """Creates the Inception Resnet V2 model.
#
#   Args:
#     inputs: a 4-D tensor of size [batch_size, height, width, 3].
#     num_classes: number of predicted classes.
#     is_training: whether is training or not.
#     dropout_keep_prob: float, the fraction to keep before final layer.
#     reuse: whether or not the network and its variables should be reused. To be
#       able to reuse 'scope' must be given.
#     scope: Optional variable_scope.
#
#   Returns:
#     logits: the logits outputs of the model.
#     end_points: the set of end_points from the inception model.
#   """
#   end_points = {}
#
#   with tf.variable_scope(scope, 'InceptionResnetV2', [inputs], reuse=reuse):
#     with slim.arg_scope([slim.batch_norm, slim.dropout],
#                         is_training=is_training):
#       with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
#                           stride=1, padding='SAME'):
#
#         # 149 x 149 x 32
#         net = slim.conv2d(inputs, 32, 3, stride=2, padding='VALID',
#                           scope='Conv2d_1a_3x3')
#         end_points['Conv2d_1a_3x3'] = net
#         _activation_summary(net)
#         # 147 x 147 x 32
#         net = slim.conv2d(net, 32, 3, padding='VALID',
#                           scope='Conv2d_2a_3x3')
#         end_points['Conv2d_2a_3x3'] = net
#         _activation_summary(net)
#         # 147 x 147 x 64
#         net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
#         end_points['Conv2d_2b_3x3'] = net
#         _activation_summary(net)
#         # 73 x 73 x 64
#         net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
#                               scope='MaxPool_3a_3x3')
#         end_points['MaxPool_3a_3x3'] = net
#         _activation_summary(net)
#         # 73 x 73 x 80
#         net = slim.conv2d(net, 80, 1, padding='VALID',
#                           scope='Conv2d_3b_1x1')
#         end_points['Conv2d_3b_1x1'] = net
#         _activation_summary(net)
#         # 71 x 71 x 192
#         net = slim.conv2d(net, 192, 3, padding='VALID',
#                           scope='Conv2d_4a_3x3')
#         end_points['Conv2d_4a_3x3'] = net
#         _activation_summary(net)
#         # 35 x 35 x 192
#         net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
#                               scope='MaxPool_5a_3x3')
#         end_points['MaxPool_5a_3x3'] = net
#         _activation_summary(net)
#
#         # 35 x 35 x 320
#         with tf.variable_scope('Mixed_5b'):
#           with tf.variable_scope('Branch_0'):
#             tower_conv = slim.conv2d(net, 96, 1, scope='Conv2d_1x1')
#           with tf.variable_scope('Branch_1'):
#             tower_conv1_0 = slim.conv2d(net, 48, 1, scope='Conv2d_0a_1x1')
#             tower_conv1_1 = slim.conv2d(tower_conv1_0, 64, 5,
#                                         scope='Conv2d_0b_5x5')
#           with tf.variable_scope('Branch_2'):
#             tower_conv2_0 = slim.conv2d(net, 64, 1, scope='Conv2d_0a_1x1')
#             tower_conv2_1 = slim.conv2d(tower_conv2_0, 96, 3,
#                                         scope='Conv2d_0b_3x3')
#             tower_conv2_2 = slim.conv2d(tower_conv2_1, 96, 3,
#                                         scope='Conv2d_0c_3x3')
#           with tf.variable_scope('Branch_3'):
#             tower_pool = slim.avg_pool2d(net, 3, stride=1, padding='SAME',
#                                          scope='AvgPool_0a_3x3')
#             tower_pool_1 = slim.conv2d(tower_pool, 64, 1,
#                                        scope='Conv2d_0b_1x1')
#           net = tf.concat(axis=3, values=[tower_conv, tower_conv1_1,
#                               tower_conv2_2, tower_pool_1])
#
#         end_points['Mixed_5b'] = net
#         net = slim.repeat(net, 10, block35, scale=0.17)
#         _activation_summary(net)
#
#         # 17 x 17 x 1088
#         with tf.variable_scope('Mixed_6a'):
#           with tf.variable_scope('Branch_0'):
#             tower_conv = slim.conv2d(net, 384, 3, stride=2, padding='VALID',
#                                      scope='Conv2d_1a_3x3')
#           with tf.variable_scope('Branch_1'):
#             tower_conv1_0 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
#             tower_conv1_1 = slim.conv2d(tower_conv1_0, 256, 3,
#                                         scope='Conv2d_0b_3x3')
#             tower_conv1_2 = slim.conv2d(tower_conv1_1, 384, 3,
#                                         stride=2, padding='VALID',
#                                         scope='Conv2d_1a_3x3')
#           with tf.variable_scope('Branch_2'):
#             tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
#                                          scope='MaxPool_1a_3x3')
#           net = tf.concat(axis=3, values=[tower_conv, tower_conv1_2, tower_pool])
#
#         end_points['Mixed_6a'] = net
#         net = slim.repeat(net, 20, block17, scale=0.10)
#         _activation_summary(net)
#
#         # Auxiliary tower
#         with tf.variable_scope('AuxLogits'):
#           aux = slim.avg_pool2d(net, 5, stride=3, padding='VALID',
#                                 scope='Conv2d_1a_3x3')
#           aux = slim.conv2d(aux, 128, 1, scope='Conv2d_1b_1x1')
#           aux = slim.conv2d(aux, 768, aux.get_shape()[1:3],
#                             padding='VALID', scope='Conv2d_2a_5x5')
#           aux = slim.flatten(aux)
#           aux = slim.fully_connected(aux, num_classes, activation_fn=None,
#                                      scope='Logits')
#           end_points['AuxLogits'] = aux
#           _activation_summary(aux)
#
#         with tf.variable_scope('Mixed_7a'):
#           with tf.variable_scope('Branch_0'):
#             tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
#             tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
#                                        padding='VALID', scope='Conv2d_1a_3x3')
#           with tf.variable_scope('Branch_1'):
#             tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
#             tower_conv1_1 = slim.conv2d(tower_conv1, 288, 3, stride=2,
#                                         padding='VALID', scope='Conv2d_1a_3x3')
#           with tf.variable_scope('Branch_2'):
#             tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
#             tower_conv2_1 = slim.conv2d(tower_conv2, 288, 3,
#                                         scope='Conv2d_0b_3x3')
#             tower_conv2_2 = slim.conv2d(tower_conv2_1, 320, 3, stride=2,
#                                         padding='VALID', scope='Conv2d_1a_3x3')
#           with tf.variable_scope('Branch_3'):
#             tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
#                                          scope='MaxPool_1a_3x3')
#           net = tf.concat(axis=3, values=[tower_conv_1, tower_conv1_1,
#                               tower_conv2_2, tower_pool])
#
#         end_points['Mixed_7a'] = net
#
#         net = slim.repeat(net, 9, block8, scale=0.20)
#         net = block8(net, activation_fn=None)
#
#         net = slim.conv2d(net, 1536, 1, scope='Conv2d_7b_1x1')
#         end_points['Conv2d_7b_1x1'] = net
#         _activation_summary(net)
#
#         with tf.variable_scope('Logits'):
#           end_points['PrePool'] = net
#           net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
#                                 scope='AvgPool_1a_8x8')
#           net = slim.flatten(net)
#
#           net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
#                              scope='Dropout')
#
#           end_points['PreLogitsFlatten'] = net
#           logits = slim.fully_connected(net, num_classes, activation_fn=None,
#                                         scope='Logits')
#           end_points['Logits'] = logits
#           end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')
#           _activation_summary(logits)
#           _activation_summary(end_points['Predictions'])
#
#     return logits, end_points


def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 35x35 resnet block."""
  with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
    with tf.variable_scope('Branch_2'):
      tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
      tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, 3, scope='Conv2d_0b_3x3')
      tower_conv2_2 = slim.conv2d(tower_conv2_1, 64, 3, scope='Conv2d_0c_3x3')
    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_1, tower_conv2_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')
    net += scale * up
    if activation_fn:
      net = activation_fn(net)
  return net


def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 17x17 resnet block."""
  with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 160, [1, 7],
                                  scope='Conv2d_0b_1x7')
      tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [7, 1],
                                  scope='Conv2d_0c_7x1')
    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')
    net += scale * up
    if activation_fn:
      net = activation_fn(net)
  return net


def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 8x8 resnet block."""
  with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 224, [1, 3],
                                  scope='Conv2d_0b_1x3')
      tower_conv1_2 = slim.conv2d(tower_conv1_1, 256, [3, 1],
                                  scope='Conv2d_0c_3x1')
    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')
    net += scale * up
    if activation_fn:
      net = activation_fn(net)
  return net


def inception_resnet_v2_base(inputs,
                             final_endpoint='Conv2d_7b_1x1',
                             output_stride=16,
                             align_feature_maps=False,
                             scope=None):
  """Inception model from  http://arxiv.org/abs/1602.07261.
  Constructs an Inception Resnet v2 network from inputs to the given final
  endpoint. This method can construct the network up to the final inception
  block Conv2d_7b_1x1.
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
      'MaxPool_3a_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'MaxPool_5a_3x3',
      'Mixed_5b', 'Mixed_6a', 'PreAuxLogits', 'Mixed_7a', 'Conv2d_7b_1x1']
    output_stride: A scalar that specifies the requested ratio of input to
      output spatial resolution. Only supports 8 and 16.
    align_feature_maps: When true, changes all the VALID paddings in the network
      to SAME padding so that the feature maps are aligned.
    scope: Optional variable_scope.
  Returns:
    tensor_out: output tensor corresponding to the final_endpoint.
    end_points: a set of activations for external use, for example summaries or
                losses.
  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
      or if the output_stride is not 8 or 16, or if the output_stride is 8 and
      we request an end point after 'PreAuxLogits'.
  """
  if output_stride != 8 and output_stride != 16:
    raise ValueError('output_stride must be 8 or 16.')

  padding = 'SAME' if align_feature_maps else 'VALID'

  end_points = {}

  def add_and_check_final(name, net):
    end_points[name] = net
    return name == final_endpoint

  with tf.variable_scope(scope, 'InceptionResnetV2', [inputs]):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        stride=1, padding='SAME'):
      # 149 x 149 x 32
      net = slim.conv2d(inputs, 32, 3, stride=2, padding=padding,
                        scope='Conv2d_1a_3x3')
      if add_and_check_final('Conv2d_1a_3x3', net): return net, end_points
      _activation_summary(net)


      # 147 x 147 x 32
      net = slim.conv2d(net, 32, 3, padding=padding,
                        scope='Conv2d_2a_3x3')
      if add_and_check_final('Conv2d_2a_3x3', net): return net, end_points
      _activation_summary(net)


      # 147 x 147 x 64
      net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
      if add_and_check_final('Conv2d_2b_3x3', net): return net, end_points
      _activation_summary(net)

      # 73 x 73 x 64
      net = slim.max_pool2d(net, 3, stride=2, padding=padding,
                            scope='MaxPool_3a_3x3')
      if add_and_check_final('MaxPool_3a_3x3', net): return net, end_points
      _activation_summary(net)

      # 73 x 73 x 80
      net = slim.conv2d(net, 80, 1, padding=padding,
                        scope='Conv2d_3b_1x1')
      if add_and_check_final('Conv2d_3b_1x1', net): return net, end_points
      _activation_summary(net)

      # 71 x 71 x 192
      net = slim.conv2d(net, 192, 3, padding=padding,
                        scope='Conv2d_4a_3x3')
      if add_and_check_final('Conv2d_4a_3x3', net): return net, end_points
      _activation_summary(net)

      # 35 x 35 x 192
      net = slim.max_pool2d(net, 3, stride=2, padding=padding,
                            scope='MaxPool_5a_3x3')
      if add_and_check_final('MaxPool_5a_3x3', net): return net, end_points
      _activation_summary(net)


      # 35 x 35 x 320
      with tf.variable_scope('Mixed_5b'):
        with tf.variable_scope('Branch_0'):
          tower_conv = slim.conv2d(net, 96, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
          tower_conv1_0 = slim.conv2d(net, 48, 1, scope='Conv2d_0a_1x1')
          tower_conv1_1 = slim.conv2d(tower_conv1_0, 64, 5,
                                      scope='Conv2d_0b_5x5')
        with tf.variable_scope('Branch_2'):
          tower_conv2_0 = slim.conv2d(net, 64, 1, scope='Conv2d_0a_1x1')
          tower_conv2_1 = slim.conv2d(tower_conv2_0, 96, 3,
                                      scope='Conv2d_0b_3x3')
          tower_conv2_2 = slim.conv2d(tower_conv2_1, 96, 3,
                                      scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          tower_pool = slim.avg_pool2d(net, 3, stride=1, padding='SAME',
                                       scope='AvgPool_0a_3x3')
          tower_pool_1 = slim.conv2d(tower_pool, 64, 1,
                                     scope='Conv2d_0b_1x1')
        net = tf.concat(
            [tower_conv, tower_conv1_1, tower_conv2_2, tower_pool_1], 3)
        _activation_summary(net)


      if add_and_check_final('Mixed_5b', net): return net, end_points
      # TODO(alemi): Register intermediate endpoints
      net = slim.repeat(net, 10, block35, scale=0.17)

      # 17 x 17 x 1088 if output_stride == 8,
      # 33 x 33 x 1088 if output_stride == 16
      use_atrous = output_stride == 8

      with tf.variable_scope('Mixed_6a'):
        with tf.variable_scope('Branch_0'):
          tower_conv = slim.conv2d(net, 384, 3, stride=1 if use_atrous else 2,
                                   padding=padding,
                                   scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
          tower_conv1_0 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
          tower_conv1_1 = slim.conv2d(tower_conv1_0, 256, 3,
                                      scope='Conv2d_0b_3x3')
          tower_conv1_2 = slim.conv2d(tower_conv1_1, 384, 3,
                                      stride=1 if use_atrous else 2,
                                      padding=padding,
                                      scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_2'):
          tower_pool = slim.max_pool2d(net, 3, stride=1 if use_atrous else 2,
                                       padding=padding,
                                       scope='MaxPool_1a_3x3')
        net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)
        _activation_summary(net)


      if add_and_check_final('Mixed_6a', net): return net, end_points

      # TODO(alemi): register intermediate endpoints
      with slim.arg_scope([slim.conv2d], rate=2 if use_atrous else 1):
        net = slim.repeat(net, 20, block17, scale=0.10)
      if add_and_check_final('PreAuxLogits', net): return net, end_points

      if output_stride == 8:
        # TODO(gpapan): Properly support output_stride for the rest of the net.
        raise ValueError('output_stride==8 is only supported up to the '
                         'PreAuxlogits end_point for now.')

      # 8 x 8 x 2080
      with tf.variable_scope('Mixed_7a'):
        with tf.variable_scope('Branch_0'):
          tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
          tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
                                     padding=padding,
                                     scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
          tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
          tower_conv1_1 = slim.conv2d(tower_conv1, 288, 3, stride=2,
                                      padding=padding,
                                      scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_2'):
          tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
          tower_conv2_1 = slim.conv2d(tower_conv2, 288, 3,
                                      scope='Conv2d_0b_3x3')
          tower_conv2_2 = slim.conv2d(tower_conv2_1, 320, 3, stride=2,
                                      padding=padding,
                                      scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_3'):
          tower_pool = slim.max_pool2d(net, 3, stride=2,
                                       padding=padding,
                                       scope='MaxPool_1a_3x3')
        net = tf.concat(
            [tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool], 3)
        _activation_summary(net)


      if add_and_check_final('Mixed_7a', net): return net, end_points

      # TODO(alemi): register intermediate endpoints
      net = slim.repeat(net, 9, block8, scale=0.20)
      net = block8(net, activation_fn=None)

      # 8 x 8 x 1536
      net = slim.conv2d(net, 1536, 1, scope='Conv2d_7b_1x1')
      if add_and_check_final('Conv2d_7b_1x1', net): return net, end_points
      _activation_summary(net)

    raise ValueError('final_endpoint (%s) not recognized', final_endpoint)


def inference(inputs, num_classes=1001, is_training=True,
                        dropout_keep_prob=0.8,
                        reuse=None,
                        scope='InceptionResnetV2',
                        create_aux_logits=True):
  """Creates the Inception Resnet V2 model.
  Args:
    inputs: a 4-D tensor of size [batch_size, height, width, 3].
    num_classes: number of predicted classes.
    is_training: whether is training or not.
    dropout_keep_prob: float, the fraction to keep before final layer.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
    create_aux_logits: Whether to include the auxilliary logits.
  Returns:
    logits: the logits outputs of the model.
    end_points: the set of end_points from the inception model.
  """
  end_points = {}

  with tf.variable_scope(scope, 'InceptionResnetV2', [inputs, num_classes],
                         reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):

      net, end_points = inception_resnet_v2_base(inputs, scope=scope)

      if create_aux_logits:
        with tf.variable_scope('AuxLogits'):
          aux = end_points['PreAuxLogits']
          aux = slim.avg_pool2d(aux, 5, stride=3, padding='VALID',
                                scope='Conv2d_1a_3x3')
          aux = slim.conv2d(aux, 128, 1, scope='Conv2d_1b_1x1')
          aux = slim.conv2d(aux, 768, aux.get_shape()[1:3],
                            padding='VALID', scope='Conv2d_2a_5x5')
          aux = slim.flatten(aux)
          aux = slim.fully_connected(aux, num_classes, activation_fn=None,
                                     scope='Logits')
          end_points['AuxLogits'] = aux
          _activation_summary(aux)

      with tf.variable_scope('Logits'):
        net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                              scope='AvgPool_1a_8x8')
        net = slim.flatten(net)

        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='Dropout')

        end_points['PreLogitsFlatten'] = net
        logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                      scope='Logits')
        end_points['Logits'] = logits
        end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')
        _activation_summary(logits)

    return logits, end_points



def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Reshape the labels into a dense Tensor of
  # shape [batch_size, NUM_CLASSES].
  sparse_labels = tf.reshape(labels, [FLAGS.batch_size, 1])
  # indices = tf.reshape(tfrange(FLAGS.batch_size), [FLAGS.batch_size, 1])
  indices = tf.reshape(range(FLAGS.batch_size), [FLAGS.batch_size, 1])
  concated = tf.concat([indices, sparse_labels],1)
  dense_labels = tf.sparse_to_dense(concated,
                                    [FLAGS.batch_size, NUM_CLASSES],
                                    1.0, 0.0)

  # Calculate the average cross entropy loss across the batch.
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=dense_labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name +' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
      if update_ops:
          updates = tf.group(*update_ops)
          total_loss = control_flow_ops.with_dependencies([updates], total_loss)

  optimizer = tf.train.GradientDescentOptimizer(lr)
  train_step = slim.learning.create_train_op(total_loss, optimizer, global_step=global_step)

  #      opt = tf.train.GradientDescentOptimizer(lr)
  #      grads = opt.compute_gradients(total_loss)
  # #
  # # Apply gradients.
  # apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
  #
  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  # for grad, var in grads:
  #   if grad is not None:
  #     tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([train_step, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [IMAGE_SIZE, IMAGE_SIZE, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    return img, label

def distort_inputs_train(img,label):
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    img = tf.image.per_image_standardization(img)
    images, labels = tf.train.shuffle_batch([img, label],
                                            batch_size=FLAGS.batch_size, capacity=min_queue_examples + 3 * FLAGS.batch_size,
                                            min_after_dequeue=min_queue_examples)
    return images, labels

def distort_inputs_train_deal(img,label):
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(img)

    # Because these operations are not commutative, consider randomizing
    # randomize the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    img = tf.image.per_image_standardization(distorted_image)

    images, labels = tf.train.shuffle_batch([img, label],
                                            batch_size=FLAGS.batch_size,
                                            capacity=min_queue_examples + 3 * FLAGS.batch_size,
                                            min_after_dequeue=min_queue_examples)
    return images, labels




# def maybe_download_and_extract():
#   """Download and extract the tarball from Alex's website."""
#   dest_directory = FLAGS.data_dir
#   if not os.path.exists(dest_directory):
#     os.mkdir(dest_directory)
#   filename = DATA_URL.split('/')[-1]
#   filepath = os.path.join(dest_directory, filename)
#   if not os.path.exists(filepath):
#     def _progress(count, block_size, total_size):
#       sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
#           float(count * block_size) / float(total_size) * 100.0))
#       sys.stdout.flush()
#     filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath,
#                                              reporthook=_progress)
#     print()
#     statinfo = os.stat(filepath)
#     print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
#     tarfile.open(filepath, 'r:gz').extractall(dest_directory)
