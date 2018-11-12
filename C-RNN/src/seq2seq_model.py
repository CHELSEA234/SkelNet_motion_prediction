
"""Sequence-to-sequence model for human motion prediction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs

import random

import numpy as np
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import data_utils
import tensorflow.contrib.layers as tcl

from spatial_decoder import AEDecoder

class Seq2SeqModel(object):
  """Sequence-to-sequence model for human motion prediction"""

  # rnn_size is hidden layer size
  def __init__(self, architecture, source_seq_len, target_seq_len, rnn_size, num_layers, max_gradient_norm, batch_size, learning_rate, 
    learning_rate_decay_factor, loss_to_use, optimizer_to_use, number_of_actions, cmu_data, alpha, beta, gamma, x_s,
    one_hot=True, residual_velocities=False, d_layers=2, dtype=tf.float32):

    if not cmu_data:
      self.HUMAN_SIZE = 54    # maybe different dataset would give different human size
    else:
      self.HUMAN_SIZE = 62    # this is from .asf, 70 is from .bvh

    self.input_size = self.HUMAN_SIZE + number_of_actions if one_hot else self.HUMAN_SIZE

    print( "One hot is ", one_hot )
    print( "Input size is %d" % self.input_size )

    self.decoder = AEDecoder(0.01, d_layers, self.HUMAN_SIZE)
    self.source_seq_len = source_seq_len
    self.target_seq_len = target_seq_len
    self.rnn_size = rnn_size
    self.batch_size = batch_size
    self.learning_rate = tf.Variable( float(learning_rate), trainable=False, dtype=dtype )
    self.sampling_rate = tf.placeholder(dtype=dtype, shape=()) 

    # === Decay ===
    self.learning_rate_decay_op = self.learning_rate.assign( self.learning_rate * learning_rate_decay_factor )
    self.global_step = tf.Variable(0, trainable=False)
    self.x_s = x_s
    print ("the mode is: ", x_s)


   # === Transform the inputs ===   
    with tf.name_scope("inputs"):  
      enc_in = tf.placeholder(dtype, shape=[None, source_seq_len-1, self.input_size], name="enc_in")
      dec_in = tf.placeholder(dtype, shape=[None, target_seq_len, self.input_size], name="dec_in")
      dec_out = tf.placeholder(dtype, shape=[None, target_seq_len, self.input_size], name="dec_out")

      self.encoder_inputs = enc_in
      self.decoder_inputs = dec_in
      self.decoder_outputs = dec_out

      enc_in = tf.transpose(enc_in, [1, 0, 2])
      dec_in = tf.transpose(dec_in, [1, 0, 2])
      dec_out = tf.transpose(dec_out, [1, 0, 2])     
    
      enc_in = tf.reshape(enc_in, [-1, self.input_size])
      dec_in = tf.reshape(dec_in, [-1, self.input_size])
      dec_out = tf.reshape(dec_out, [-1, self.input_size])

      enc_in = tf.split(enc_in, source_seq_len-1, axis=0)
      dec_in = tf.split(dec_in, target_seq_len, axis=0)
      dec_out = tf.split(dec_out, target_seq_len, axis=0)

    self.is_training = tf.placeholder(tf.bool)
    """arrange cell and transform the input"""
    only_cell   = tf.contrib.rnn.GRUCell( self.rnn_size )


    print (len(enc_in), enc_in[0].get_shape())

    for index, item in enumerate(enc_in):
      if index == 0:
        enc_in_list = item
      else:
        enc_in_list = tf.concat([enc_in_list, item], axis=1)
    enc_in_list = tf.concat([enc_in_list, dec_in[0]], axis=1)

    outputs     = []
    outputs_GT  = []
    outputs_s   = []

    """define loss function and architecture type"""

    def lf(prev, i):
      return prev

    def lrelu(x, leak=0.2, name="lrelu"):
      return tf.maximum(x, leak*x)

    sp_decoder=self.decoder
    loop_function = lf

    only_cell = core_rnn_cell.OutputProjectionWrapper(only_cell, self.rnn_size)  
    output_size = self.rnn_size

    batch_size = array_ops.shape(dec_in[0])[0]
    state = only_cell.zero_state(batch_size=batch_size, dtype=dtype)
    initial_state = state
    state_GT = state
    state_t  = state
    keep_prob_ = 0.8
    initializer_weight = tf.random_uniform_initializer(minval=-0.04, maxval=0.04)
    initializer_bias = tf.random_uniform_initializer(minval=-0.04, maxval=0.04)

    def my_drop_out(output):
      return tf.where(self.is_training, tcl.dropout(output, keep_prob = keep_prob_, is_training=True), output)

    def my_fc(input_, output, scope, reuse=None):
      return tcl.fully_connected(input_, output, scope=scope, activation_fn=None, weights_initializer=initializer_weight, 
        biases_initializer=initializer_bias, reuse=reuse)

    dim_1 = 128
    dim_2 = 256

    with vs.variable_scope("attention_decoder", dtype=dtype) as scope:
      prev    = None

      for i, inp in enumerate(dec_in):
        if i > 0:
          vs.get_variable_scope().reuse_variables()

        inp_GT  = inp

        if loop_function is not None and prev is not None:
          with vs.variable_scope("loop_function", reuse=True):
            inp   = lf(prev, i)         # inp is for T-RNN

        with vs.variable_scope("RNN"):
          cell_output, state        = only_cell(inp, state)
          vs.get_variable_scope().reuse_variables()
          cell_output_GT, state_GT  = only_cell(inp_GT, state_GT)

        with vs.variable_scope("t_decoder"):
          clear_output = sp_decoder.forward(single_input = cell_output, is_training = self.is_training)
          output       = clear_output + inp
          vs.get_variable_scope().reuse_variables()
          clear_output_GT = sp_decoder.forward(single_input = cell_output_GT, is_training = self.is_training)
          output_GT       = clear_output_GT + inp_GT

        prev    = output 
        outputs.append(output)
        outputs_GT.append(output_GT) 

    self.outputs      = outputs             # GX: temprol output
    self.dec_out      = dec_out

    # GX: scope mechanism: Decoder's w1 and d1, Encoder's w1 and d1
    with tf.name_scope("loss_angles"): 
      loss_angles    = tf.reduce_mean(tf.square(tf.subtract(dec_out, outputs)))      
      loss_angles_GT = tf.reduce_mean(tf.square(tf.subtract(dec_out, outputs_GT))) 

    self.loss_t      = alpha * loss_angles_GT + beta * loss_angles
    print ('current alpha and beta are: ', alpha, beta) 
    self.loss_summary = tf.summary.scalar('loss/loss', self.loss_t)

    """regularizers"""
    params = tf.trainable_variables()
    opt_t  = tf.train.AdamOptimizer(self.learning_rate)   
    RNN_var = [var_ for var_ in params if "RNN" in var_.name]
    t_dec_var = [var_ for var_ in params if "t_decoder" in var_.name]

    print ("================= variable ===================================")
    for reg_var in params:
      shp = reg_var.get_shape().as_list()
      print("- {} shape:{} size:{}".format(reg_var.name, shp, np.prod(shp)))
    print ("=================TRNN variable (loss_t)=======================")
    for reg_var in RNN_var:
      shp = reg_var.get_shape().as_list()
      print("- {} shape:{} size:{}".format(reg_var.name, shp, np.prod(shp)))
    print ("================T-decoder variable (loss_t)===================")
    self.reg_t = 0
    scale = 0.01
    count = 0
    for reg_var in t_dec_var:
      shp = reg_var.get_shape().as_list()
      print("- {} shape:{} size:{}".format(reg_var.name, shp, np.prod(shp)))
      count = count + np.prod(shp)
      self.reg_t = self.reg_t + scale*tf.nn.l2_loss(reg_var)
    print ("total number is ", count)
    print ("the scale is ", scale)
    print ("===========================================")   

    self.loss_t = self.loss_t + self.reg_t

    gradients_t   = tf.gradients( self.loss_t,  RNN_var + t_dec_var )
    clipped_gradients_t, self.gradient_norms   = tf.clip_by_global_norm(gradients_t, max_gradient_norm)
    self.updates_t  = opt_t.apply_gradients(zip(clipped_gradients_t, RNN_var + t_dec_var), global_step=self.global_step)
    self.updates   = self.updates_t


    # Keep track of the learning rate
    self.learning_rate_summary = tf.summary.scalar('learning_rate/learning_rate', self.learning_rate)
    self.saver = tf.train.Saver( tf.global_variables(), max_to_keep=10000 )           # better for drawing plot

  def step_train_t(self, is_training, session, encoder_inputs, decoder_inputs, decoder_outputs, sampling_rate=None):
    input_feed = {self.encoder_inputs: encoder_inputs, self.decoder_inputs: decoder_inputs, self.decoder_outputs: decoder_outputs, 
      self.is_training: is_training, self.sampling_rate: sampling_rate}
    output_feed = [self.updates, self.gradient_norms, self.loss_t] 

    outputs = session.run(output_feed, input_feed)
    return outputs[0], outputs[1], outputs[2]

  def step_test(self, is_training, session, encoder_inputs, decoder_inputs, decoder_outputs, sampling_rate=None):
    input_feed = {self.encoder_inputs: encoder_inputs, self.decoder_inputs: decoder_inputs, self.decoder_outputs: decoder_outputs, 
      self.is_training: is_training, self.sampling_rate: sampling_rate}
    output_feed = [self.loss_t, self.outputs, self.outputs, self.outputs, self.loss_summary]

    outputs = session.run(output_feed, input_feed)
    return outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]

  def get_batch( self, data ):
    """Get a random batch of data from the specified bucket, prepare for step.
      GX: a random batch of data to train
    Args
      data: a list of sequences of size n-by-d to fit the model to.
      actions: a list of the actions we are using
    Returns
      The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
      the constructed batches have the proper format to call step(...) later.
    """

    # Select entries at random
    all_keys    = list(data.keys())
    # print (len(all_keys))
    chosen_keys = np.random.choice( len(all_keys), self.batch_size )
    # print ('chosen_keys is: ', chosen_keys)   #this ensures getting sequence randomly

    # How many frames in total do we need?
    total_frames = self.source_seq_len + self.target_seq_len

    encoder_inputs  = np.zeros((self.batch_size, self.source_seq_len-1, self.input_size), dtype=float)
    decoder_inputs  = np.zeros((self.batch_size, self.target_seq_len, self.input_size), dtype=float)
    decoder_outputs = np.zeros((self.batch_size, self.target_seq_len, self.input_size), dtype=float)

    for i in xrange( self.batch_size ):

      the_key = all_keys[ chosen_keys[i] ]

      # Get the number of frames
      n, _ = data[ the_key ].shape

      # Sample somewherein the middle
      idx = np.random.randint( 16, n-total_frames )

      # print (the_key, idx)

      # Select the data around the sampled points
      data_sel = data[ the_key ][idx:idx+total_frames ,:]

      # Add the data
      encoder_inputs[i,:,0:self.input_size]  = data_sel[0:self.source_seq_len-1, :]
      decoder_inputs[i,:,0:self.input_size]  = data_sel[self.source_seq_len-1:self.source_seq_len+self.target_seq_len-1, :]
      decoder_outputs[i,:,0:self.input_size] = data_sel[self.source_seq_len:, 0:self.input_size]

    # print (encoder_inputs.shape, decoder_inputs.shape, decoder_outputs.shape)
    return encoder_inputs, decoder_inputs, decoder_outputs

  def find_indices_srnn( self, data, action, batch_size, subject = 5 ):     # here subject is 5 in test, randomly chosen in validation
    """
    Find the same action indices as in SRNN.
    See https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
    """

    # Used a fixed dummy seed, following
    # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
    SEED = 1234567890
    rng = np.random.RandomState( SEED )

    subaction1 = 1
    subaction2 = 2

    # print ((subject, action, subaction1, 'even'))
    T1 = data[ (subject, action, subaction1, 'even') ].shape[0]
    T2 = data[ (subject, action, subaction2, 'even') ].shape[0]
    prefix, suffix = 50, 100

    idx = []
    for i in range(int(batch_size/2)):
      idx.append( rng.randint( 16,T1-prefix-suffix ))
      idx.append( rng.randint( 16,T2-prefix-suffix ))
    # print (idx)
    return idx

  def get_batch_srnn(self, data, action, batch_size = 8, phase = 'test', val_subject = None):
    """
    Get a random batch of data from the specified bucket, prepare for step.
    GX: Get a random batch of data sequence of 50 input and 25 output, based on random seed 
    Args
      data: dictionary with k:v, k=((subject, action, subsequence, 'even')),
        v=nxd matrix with a sequence of poses
      action: the action to load data from
    Returns
      The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
      the constructed batches have the proper format to call step(...) later.
    """

    actions = ["directions", "discussion", "eating", "greeting", "phoning",
              "posing", "purchases", "sitting", "sittingdown", "smoking",
              "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]

    if not action in actions:
      raise ValueError("Unrecognized action {0}".format(action))

    frames = {}
    if phase != 'test' :     # for validation, this is for different subject number
      subject = val_subject
      frames[ action ] = self.find_indices_srnn( data, action, batch_size, subject)
    else:
      subject = 5
      frames[ action ] = self.find_indices_srnn( data, action, batch_size )
    # so far, you get 8 frames
    # print (frames[action])  # [1087, 955, 1145, 332, 660, 304, 201, 54]; 
    # [1087, 955, 1145, 332, 660, 304, 201, 54, 1076, 335, 1340, 335, 1323, 446, 1251, 1093, 714, 123, 695, 33, 90, 509, 367, 714, 120, 730, 881, 53, 728, 631, 443, 555]

    batch_size = batch_size   # we always evaluate 8 seeds at fixed mode, larger number at true vaildation case
    # subject    = 5    # we always evaluate on subject 5, GX: modify it for validation and test
    source_seq_len = self.source_seq_len
    target_seq_len = self.target_seq_len

    seeds = [( action, (i%2)+1, frames[action][i] ) for i in range(batch_size)]
    # print (seeds) # [('walking', 1, 1087), ('walking', 2, 955), ('walking', 1, 1145), ('walking', 2, 332), ('walking', 1, 660), ('walking', 2, 304), ('walking', 1, 201), ('walking', 2, 54)]

    encoder_inputs  = np.zeros( (batch_size, source_seq_len-1, self.input_size), dtype=float )
    decoder_inputs  = np.zeros( (batch_size, target_seq_len, self.input_size), dtype=float )
    decoder_outputs = np.zeros( (batch_size, target_seq_len, self.input_size), dtype=float )

    # Compute the number of frames needed
    total_frames = source_seq_len + target_seq_len
    # print (total_frames)

    # Reproducing SRNN's sequence subsequence selection as done in
    # https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L343
    for i in xrange( batch_size ):

      _, subsequence, idx = seeds[i]
      idx = idx + 50    # you already subtract 50 before in find_indices_srnn

      data_sel = data[ (subject, action, subsequence, 'even') ]

      data_sel = data_sel[(idx-source_seq_len):(idx+target_seq_len) ,:]
      # print (idx, subsequence, data_sel.shape)  # 1137 1 (75, 55), 1005 2 (75, 55), 1195 1 (75, 55)
      encoder_inputs[i, :, :]  = data_sel[0:source_seq_len-1, :]
      decoder_inputs[i, :, :]  = data_sel[source_seq_len-1:(source_seq_len+target_seq_len-1), :]
      decoder_outputs[i, :, :] = data_sel[source_seq_len:, :]

    # print (encoder_inputs.shape, decoder_inputs.shape, decoder_outputs.shape, source_seq_len-1, (source_seq_len+target_seq_len-1), target_seq_len)
    # (8, 49, 55) (8, 25, 55) (8, 25, 55) 49 74
    return encoder_inputs, decoder_inputs, decoder_outputs

