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
    learning_rate_decay_factor, summaries_dir, loss_to_use, optimizer_to_use, number_of_actions, cmu_data, long_t, alpha, beta, gamma, x_s,
    one_hot=True, residual_velocities=False, d_layers=2, dtype=tf.float32):

    if not cmu_data:
      self.HUMAN_SIZE = 54    # maybe different dataset would give different human size
    else:
      self.HUMAN_SIZE = 62    # this is from .asf, 70 is from .bvh

    self.input_size = self.HUMAN_SIZE + number_of_actions if one_hot else self.HUMAN_SIZE

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

    for index, item in enumerate(enc_in):
      if index == 0:
        enc_in_list = item
      else:
        enc_in_list = tf.concat([enc_in_list, item], axis=1)
    enc_in_list = tf.concat([enc_in_list, dec_in[0]], axis=1)

    outputs     = []

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

        if loop_function is not None and prev is not None:
          with vs.variable_scope("loop_function", reuse=True):
            inp   = lf(prev, i)         # inp is for T-RNN

        h0 = inp
        with vs.variable_scope("s_decoder"):
          # GX: I don't why I can't have closed form of this.
          output_r_l      = my_drop_out( lrelu(  my_fc(h0[:, :14],       dim_1, scope="r_l/fc1")))
          output_l_l      = my_drop_out( lrelu(  my_fc(h0[:, 14:22],     dim_1, scope="l_l/fc1")))
          output_trunk    = my_drop_out( lrelu(  my_fc(h0[:, 22:34],     dim_1, scope="trunk/fc1")))
          output_l_u      = my_drop_out( lrelu(  my_fc(h0[:, 34:44],     dim_1, scope="l_u/fc1")))
          output_r_u      = my_drop_out( lrelu(  my_fc(h0[:, 44:54],     dim_1, scope="r_u/fc1")))
          
          output_r_l      = my_drop_out( lrelu(  my_fc(output_r_l,       dim_2, scope="r_l/fc2")))
          output_l_l      = my_drop_out( lrelu(  my_fc(output_l_l,       dim_2, scope="l_l/fc2")))
          output_trunk    = my_drop_out( lrelu(  my_fc(output_trunk,     dim_2, scope="trunk/fc2")))
          output_l_u      = my_drop_out( lrelu(  my_fc(output_l_u,       dim_2, scope="l_u/fc2")))
          output_r_u      = my_drop_out( lrelu(  my_fc(output_r_u,       dim_2, scope="r_u/fc2")))

          output_r_l      = my_drop_out( lrelu(  my_fc(output_r_l,       dim_1, scope="r_l/fc3")))
          output_l_l      = my_drop_out( lrelu(  my_fc(output_l_l,       dim_1, scope="l_l/fc3")))
          output_trunk    = my_drop_out( lrelu(  my_fc(output_trunk,     dim_1, scope="trunk/fc3")))
          output_l_u      = my_drop_out( lrelu(  my_fc(output_l_u,       dim_1, scope="l_u/fc3")))
          output_r_u      = my_drop_out( lrelu(  my_fc(output_r_u,       dim_1, scope="r_u/fc3")))       

          output_s = tf.concat([output_r_l, output_l_l, output_trunk, output_l_u, output_r_u], axis=1)

          output_s = my_fc(output_s, self.HUMAN_SIZE, scope="fc4")
          output_s = my_drop_out(lrelu(output_s))
          output_s = output_s + inp

          enc_in_list = tf.concat([enc_in_list[:, self.HUMAN_SIZE: ], output_s], axis = 1)

          output = output_s

        prev    = output 
        outputs.append(output)

    self.outputs    = outputs           # GX: used for training
    self.dec_out    = dec_out

    loss_angles     = tf.reduce_mean(tf.square(tf.subtract(dec_out, outputs)))      
    self.loss         = loss_angles
    self.loss_summary = tf.summary.scalar('loss/loss', self.loss)

    """regularizers"""
    params = tf.trainable_variables()
    opt_s  = tf.train.GradientDescentOptimizer(1e-2)  
    s_dec_var = [var_ for var_ in params if "s_decoder" in var_.name]

    print ("================= variable ===================================")
    for reg_var in params:
      shp = reg_var.get_shape().as_list()
      print("- {} shape:{} size:{}".format(reg_var.name, shp, np.prod(shp)))
    print ("===============decoder variable (loss_s)======================")
    self.reg = 0
    scale = 0.001
    count = 0
    for reg_var in s_dec_var:
      shp = reg_var.get_shape().as_list()
      print("- {} shape:{} size:{}".format(reg_var.name, shp, np.prod(shp)))
      count = count + np.prod(shp)
      self.reg = self.reg + scale*tf.nn.l2_loss(reg_var)
    print ("total number is ", count)
    print ("the scale is ", scale)
    print ("===========================================")   

    self.loss = self.loss + self.reg

    # Update all the trainable parameters
    gradients = tf.gradients( self.loss, s_dec_var )
    clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
    self.gradient_norms = norm
    self.updates = opt_s.apply_gradients(zip(clipped_gradients, s_dec_var), global_step=self.global_step)

    self.learning_rate_summary = tf.summary.scalar('learning_rate/learning_rate', self.learning_rate)
    self.saver = tf.train.Saver( tf.global_variables(), max_to_keep=10000 )           # better for drawing plot

  def step_train(self, is_training, session, encoder_inputs, decoder_inputs, decoder_outputs, sampling_rate=None):
    input_feed = {self.encoder_inputs: encoder_inputs, self.decoder_inputs: decoder_inputs, self.decoder_outputs: decoder_outputs, 
      self.is_training: is_training, self.sampling_rate: sampling_rate}
    output_feed = [self.updates, self.gradient_norms, self.loss] 

    outputs = session.run(output_feed, input_feed)
    return outputs[0], outputs[1], outputs[2]    

  def step_test(self, is_training, session, encoder_inputs, decoder_inputs, decoder_outputs, sampling_rate=None):
    input_feed = {self.encoder_inputs: encoder_inputs, self.decoder_inputs: decoder_inputs, self.decoder_outputs: decoder_outputs, 
      self.is_training: is_training, self.sampling_rate: sampling_rate}
    output_feed = [self.loss, self.outputs, self.loss_summary]

    outputs = session.run(output_feed, input_feed)
    return outputs[0], outputs[1], outputs[2]

  def get_batch( self, data, noise_rate ):
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

      # Select the data around the sampled points
      data_sel = data[ the_key ][idx:idx+total_frames ,:]

      # Add the data
      A = data_sel[0:self.source_seq_len-1, :]
      B = data_sel[self.source_seq_len-1:self.source_seq_len+self.target_seq_len-1, :]

      A_ = np.random.normal(0, noise_rate, A.shape)
      B_ = np.random.normal(0, noise_rate, B.shape)

      encoder_inputs[i,:,0:self.input_size]  = A + A_
      decoder_inputs[i,:,0:self.input_size]  = B + B_
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
    # print(idx)
    return idx

  def get_batch_srnn(self, data, action, noise_rate, batch_size = 8, phase = 'test', val_subject = None):
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
    # print (seeds) # [('walking', 1, 1087), ('walking', 2, 955), ('walking', 1, 1145), ('walking', 2, 332), ('walking', 1, 660), ('walking', 2, 304), 
    # ('walking', 1, 201), ('walking', 2, 54)]

    encoder_inputs  = np.zeros( (batch_size, source_seq_len-1, self.input_size), dtype=float )
    decoder_inputs  = np.zeros( (batch_size, target_seq_len, self.input_size), dtype=float )
    decoder_outputs = np.zeros( (batch_size, target_seq_len, self.input_size), dtype=float )

    # Compute the number of frames needed
    total_frames = source_seq_len + target_seq_len
    # print ("total_frames is: ", total_frames)
    
    SEED_n = 987654321
    rng_noise = np.random.RandomState( SEED_n)

    for i in xrange( batch_size ):

      _, subsequence, idx = seeds[i]
      idx = idx + 50    # you already subtract 50 before in find_indices_srnn

      data_sel = data[ (subject, action, subsequence, 'even') ]

      data_sel = data_sel[(idx-source_seq_len):(idx+target_seq_len) ,:]
      A = data_sel[0:source_seq_len-1, :]
      B = data_sel[source_seq_len-1:(source_seq_len+target_seq_len-1), :]

      A_ = rng_noise.normal(0, noise_rate, A.shape)
      B_ = rng_noise.normal(0, noise_rate, B.shape)
      # print(B_)

      encoder_inputs[i, :, :]  = A + A_
      decoder_inputs[i, :, :]  = B + B_
      decoder_outputs[i, :, :] = data_sel[source_seq_len:, :]

    # print (encoder_inputs.shape, decoder_inputs.shape, decoder_outputs.shape, source_seq_len-1, (source_seq_len+target_seq_len-1), target_seq_len)
    # (8, 49, 55) (8, 25, 55) (8, 25, 55) 49 74
    return encoder_inputs, decoder_inputs, decoder_outputs





