
"""Simple code for training an RNN for motion prediction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random
import time
import h5py
import os

import numpy as np
from six.moves import xrange    # pylint: disable=redefined-builtin
import tensorflow as tf
import data_utils
import matplotlib.pyplot as plt
import tensorflow.contrib.layers as tcl

from tensorflow.contrib.layers.python.layers import initializers
from data_helper import get_srnn_gts, define_actions, read_all_data
from graph_import import create_model_s, create_model_t

# Learning
tf.app.flags.DEFINE_float("learning_rate", 5e-5, "Learning rate.")    # maybe you could have a lower learning rate
tf.app.flags.DEFINE_float("sampling_rate", 1.0, "Learning rate.")
tf.app.flags.DEFINE_float("ending_sampling_rate", 0.2, "Learning rate.")
tf.app.flags.DEFINE_float("alpha", 1.0, "parameters before loss_1, loss_converge")
tf.app.flags.DEFINE_float("beta", 0.1, "parameters before loss_2, loss_reality")
tf.app.flags.DEFINE_float("gamma", 0.0, "parameters before loss_3, loss_balanced")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95, "Learning rate is multiplied by this much. 1 means no decay.")
tf.app.flags.DEFINE_integer("learning_rate_step", 10000, "Every this many steps, do decay.")    # 10000
tf.app.flags.DEFINE_float("max_gradient_norm", 5, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("iterations", int(5e4), "Iterations to train for.")   # int(5e4)  2
tf.app.flags.DEFINE_integer("data_aug", 2, "data au")

# Architecture
tf.app.flags.DEFINE_string("architecture", "tied", "Seq2seq architecture to use: [basic, tied, attention].")
tf.app.flags.DEFINE_string("opt", "SGD", "optimizer to use: [SGD, Adam].")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer, this should be half of dimensions")
tf.app.flags.DEFINE_integer("d_layers", 2, "Spatial decoder's layer number, [0, 1, 2, 3], 0 means convolutional layers")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("seq_length_in", 10, "Number of frames to feed into the encoder. 25 fps")
tf.app.flags.DEFINE_integer("seq_length_out", 10, "Number of frames that the decoder has to predict.")
tf.app.flags.DEFINE_boolean("residual_velocities", False, "Add a residual connection that effectively models velocities")

# Directories
tf.app.flags.DEFINE_string("data_dir", os.path.normpath("./data/h3.6m/dataset"), "Data directory")
tf.app.flags.DEFINE_boolean("CMU", False, "Would CMU mocap be used.")
tf.app.flags.DEFINE_string("action","all", "The action to train on. all means all the actions, all_periodic means walking, eating and smoking")
tf.app.flags.DEFINE_string("loss_to_use","sampling_based", "The type of loss to use, [sampling_based, conditioned_LSTM, scheduled_sampling, skeleton_sampling]")
tf.app.flags.DEFINE_string("x_s","inp_s", "The input for t: [inp_s, out_s]")

# remember to change test_every for quicker running time, the original value is 1000
tf.app.flags.DEFINE_integer("test_every", 1000, "How often to compute error on the test set.")      # 1000
tf.app.flags.DEFINE_integer("save_every", 50000, "How often to compute error on the test set.")     # 25000
tf.app.flags.DEFINE_integer("round", 1, "round number, need to change it manually")
tf.app.flags.DEFINE_boolean("sample", False, "Set to True for sampling.")
tf.app.flags.DEFINE_boolean("use_cpu", False, "Whether to use the CPU")
tf.app.flags.DEFINE_integer("load", 50000, "Try to load a previous checkpoint.[eating, smoking, discussion] is 29000")
tf.app.flags.DEFINE_integer("load_t", 15000, "Try to load a previous checkpoint.")

FLAGS = tf.app.flags.FLAGS
SAMPLES_FNAME = 'samples.h5'

dtype = tf.float32
keep_prob_ = 0.8

initializer = tf.random_uniform_initializer()
initializer_weight = tf.random_uniform_initializer(minval=-0.04, maxval=0.04)
initializer_bias = tf.random_uniform_initializer(minval=-0.04, maxval=0.04)

s_input_pl    = tf.placeholder(tf.float32, shape=[FLAGS.seq_length_out, None, 54], name="s_input")
t_input_pl    = tf.placeholder(tf.float32, shape=[FLAGS.seq_length_out, None, 54], name="t_input")
dec_out_pl    = tf.placeholder(tf.float32, shape=[FLAGS.seq_length_out, None, 54], name="dec_out")
is_training   = tf.placeholder(tf.bool)

def my_drop_out(output):
  return tf.where(is_training, tcl.dropout(output, keep_prob = keep_prob_, is_training=True), output)

def my_fc(input_, output, scope, reuse=None):
  return tcl.fully_connected(input_, output, scope=scope, activation_fn=None, weights_initializer=initializer_weight, 
    biases_initializer=initializer_bias, reuse=reuse)

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

s_input = s_input_pl
t_input = t_input_pl
dec_out = dec_out_pl

s_input_m   = tf.reshape(s_input,  [-1, int(FLAGS.seq_length_out * 54)])
t_input_m   = tf.reshape(t_input,  [-1, int(FLAGS.seq_length_out * 54)])
dec_out_m   = tf.reshape(dec_out,  [-1, int(FLAGS.seq_length_out * 54)])

w_mapping_s = tf.Variable( float(0.5), trainable=True, name="s_decoder/weight_s", dtype=dtype )
w_mapping_t = tf.Variable( float(0.5), trainable=True, name="s_decoder/weight_t", dtype=dtype )
w_mapping   = [w_mapping_s, w_mapping_t]


with tf.variable_scope("encoder", initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01), dtype=dtype) as scope:
  w_1  = tf.get_variable("proj_w_out_1", [540, 256])
  w_2  = tf.get_variable("proj_w_out_2", [256, 128])
  w_3  = tf.get_variable("proj_w_out_3", [128, 64])
  w_4  = tf.get_variable("proj_w_out_4", [64, 128])
  w_5  = tf.get_variable("proj_w_out_5", [128, 256])
  w_6  = tf.get_variable("proj_w_out_6", [256, 540])

with tf.variable_scope("encoder_bias", initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01), dtype=dtype) as scope: 
  b_1 = tf.get_variable("proj_b_out_1", [256])
  b_2 = tf.get_variable("proj_b_out_2", [128])
  b_3 = tf.get_variable("proj_b_out_3", [64])  
  b_4 = tf.get_variable("proj_b_out_4", [128])
  b_5 = tf.get_variable("proj_b_out_5", [256]) 
  b_6 = tf.get_variable("proj_b_out_6", [540])   

h_s         = w_mapping_s * s_input_m
h_t         = w_mapping_t * t_input_m
output_m    = tf.add(h_s, h_t)
output_m_i  = output_m

output_i = lrelu( tf.add(tf.matmul(output_m, w_1), b_1))
output_i = lrelu( tf.add(tf.matmul(output_i, w_2), b_2))
output_i = lrelu( tf.add(tf.matmul(output_i, w_3), b_3))
output_i = lrelu( tf.add(tf.matmul(output_i, w_4), b_4))
output_i = lrelu( tf.add(tf.matmul(output_i, w_5), b_5))
output_i = lrelu( tf.add(tf.matmul(output_i, w_6), b_6))
output_i = output_i + output_m


with tf.name_scope("format"):
  s_input_i = tf.reshape(s_input,  [-1, 54])   
  t_input_i = tf.reshape(t_input,  [-1, 54])
  dec_out_i = tf.reshape(dec_out,  [-1, 54])
  output_m_i = tf.reshape(output_m_i, [-1, 54])
  output_i  = tf.reshape(output_i, [-1, 54])

  s_input_i  = tf.split(s_input_i, FLAGS.seq_length_out, axis=0) 
  t_input_i  = tf.split(t_input_i, FLAGS.seq_length_out, axis=0)
  dec_out_i  = tf.split(dec_out_i, FLAGS.seq_length_out, axis=0)     
  output_m_i  = tf.split(output_m_i, FLAGS.seq_length_out, axis=0) 
  output_i   = tf.split(output_i, FLAGS.seq_length_out, axis=0)
          
loss_s = tf.reduce_mean(tf.square(tf.subtract(s_input_i, dec_out_i)))     # GX: right
loss_t = tf.reduce_mean(tf.square(tf.subtract(t_input_i, dec_out_i)))     # GX: right
loss_m = tf.reduce_mean(tf.square(tf.subtract(output_m_i, dec_out_i)))    # GX: right
loss_i = tf.reduce_mean(tf.square(tf.subtract(output_i, dec_out_i)))    # GX: right

# optimizer_m = tf.train.AdamOptimizer(1e-3).minimize(loss_m, var_list=[w_mapping_s, w_mapping_t])
# optimizer_i = tf.train.AdamOptimizer(1e-3).minimize(loss_i, var_list=[w_1, w_2, w_3, w_4, w_5, w_6, b_1, b_2, b_3, b_4, b_5, b_6])
optimizer_i = tf.train.AdamOptimizer(1e-3).minimize(loss_i, var_list=[w_1, w_2, w_3, w_4, w_5, w_6, 
                      b_1, b_2, b_3, b_4, b_5, b_6, w_mapping_s, w_mapping_t])

def batch_process(x, y, z):
  return np.hstack(x), np.hstack(y), np.hstack(z)

def angles_error(srnn_poses_s, srnn_poses_t, srnn_poses_p, srnn_poses_m, data_mean, data_std, dim_to_ignore, actions, 
  number, srnn_gts_euler, srnn_gts_expmap):

  action = actions[0]

  srnn_pred_expmap_s = data_utils.revert_output_format( srnn_poses_s, data_mean, data_std, dim_to_ignore, actions )
  srnn_pred_expmap_t = data_utils.revert_output_format( srnn_poses_t, data_mean, data_std, dim_to_ignore, actions )
  srnn_pred_expmap_p = data_utils.revert_output_format( srnn_poses_p, data_mean, data_std, dim_to_ignore, actions )
  srnn_pred_expmap_m = data_utils.revert_output_format( srnn_poses_m, data_mean, data_std, dim_to_ignore, actions )

  mean_errors_s = np.zeros( (len(srnn_pred_expmap_s), srnn_pred_expmap_s[0].shape[0]) )
  mean_errors_t = np.zeros( (len(srnn_pred_expmap_t), srnn_pred_expmap_t[0].shape[0]) )
  mean_errors_p = np.zeros( (len(srnn_pred_expmap_p), srnn_pred_expmap_p[0].shape[0]) )
  mean_errors_m = np.zeros( (len(srnn_pred_expmap_m), srnn_pred_expmap_m[0].shape[0]) )

  N_SEQUENCE_TEST = 8

  for i in np.arange(N_SEQUENCE_TEST):
    eulerchannels_pred_s = srnn_pred_expmap_s[i] 
    eulerchannels_pred_t = srnn_pred_expmap_t[i]
    eulerchannels_pred_p = srnn_pred_expmap_p[i]
    eulerchannels_pred_m = srnn_pred_expmap_m[i]        

    # Convert from exponential map to Euler angles
    for j in np.arange( eulerchannels_pred_s.shape[0] ):
      for k in np.arange(3,number,3):
        eulerchannels_pred_s[j,k:k+3] = data_utils.rotmat2euler(
          data_utils.expmap2rotmat( eulerchannels_pred_s[j,k:k+3] )) 
        eulerchannels_pred_t[j,k:k+3] = data_utils.rotmat2euler(
          data_utils.expmap2rotmat( eulerchannels_pred_t[j,k:k+3] )) 
        eulerchannels_pred_p[j,k:k+3] = data_utils.rotmat2euler(
          data_utils.expmap2rotmat( eulerchannels_pred_p[j,k:k+3] ))  
        eulerchannels_pred_m[j,k:k+3] = data_utils.rotmat2euler(
          data_utils.expmap2rotmat( eulerchannels_pred_m[j,k:k+3] ))                                                                    

    gt_i=np.copy(srnn_gts_euler[action][i])
    gt_i[:,0:6] = 0     # the translation is 0?

    idx_to_use  = np.where( np.std( gt_i, 0 ) > 1e-4 )[0] 
    euc_error_s = np.power( gt_i[:,idx_to_use] - eulerchannels_pred_s[:,idx_to_use], 2)  
    euc_error_t = np.power( gt_i[:,idx_to_use] - eulerchannels_pred_t[:,idx_to_use], 2) 
    euc_error_p = np.power( gt_i[:,idx_to_use] - eulerchannels_pred_p[:,idx_to_use], 2) 
    euc_error_m = np.power( gt_i[:,idx_to_use] - eulerchannels_pred_m[:,idx_to_use], 2)  

    euc_error_s = np.sum(euc_error_s, 1)
    euc_error_t = np.sum(euc_error_t, 1)  
    euc_error_p = np.sum(euc_error_p, 1)  
    euc_error_m = np.sum(euc_error_m, 1) 

    euc_error_s = np.sqrt( euc_error_s )
    euc_error_t = np.sqrt( euc_error_t ) 
    euc_error_p = np.sqrt( euc_error_p )
    euc_error_m = np.sqrt( euc_error_m ) 

    mean_errors_s[i,:]  = euc_error_s
    mean_errors_t[i,:]  = euc_error_t 
    mean_errors_p[i,:]  = euc_error_p
    mean_errors_m[i,:]  = euc_error_m        


  mean_mean_errors_s = np.mean( mean_errors_s, 0 )
  mean_mean_errors_t = np.mean( mean_errors_t, 0 )
  mean_mean_errors_p = np.mean( mean_errors_p, 0 )
  mean_mean_errors_m = np.mean( mean_errors_m, 0 )   

  return mean_mean_errors_s, mean_mean_errors_t, mean_mean_errors_p, mean_mean_errors_m

def train():
  """Train a seq2seq model on human motion"""

  actions = define_actions( FLAGS.action )    # here is like a list of actions, eg, ['walking'], ['walking, waiting, directions']
  action  = actions[0]

  train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = read_all_data( actions, FLAGS.seq_length_in, FLAGS.seq_length_out, FLAGS.data_dir)
  number = 97

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)        # Allowing GPU memory growth, this means whole GPU
  device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}

  with tf.Session(config=tf.ConfigProto( gpu_options=gpu_options, device_count = device_count )) as sess:    
    sess.run(tf.global_variables_initializer())

    for c_step in xrange( FLAGS.iterations ):

      if c_step == 0:
        g = tf.Graph()
        sess_1 = tf.Session(graph=g)
        with g.as_default():
          model   = create_model_s( sess_1, actions, FLAGS = FLAGS )
          print( "spatial model created" )

        h = tf.Graph()
        sess_2 = tf.Session(graph=h)
        with h.as_default():
          model_t = create_model_t( sess_2, actions, FLAGS = FLAGS )
          print( "temporal model created" ) 

      encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch( train_set )

      _, srnn_poses_s, dec_true, _ , enc_in_list   =   model.step_test(False, sess_1, encoder_inputs, decoder_inputs, decoder_outputs, sampling_rate=0.0)
      _, srnn_poses_t, _                           =   model_t.step_test(False, sess_2, encoder_inputs, decoder_inputs, decoder_outputs, sampling_rate=0.0)

      input_s, input_t, input_true = batch_process(srnn_poses_s, srnn_poses_t, dec_true)
      sess.run([optimizer_i], feed_dict={s_input_pl: np.asarray(srnn_poses_s), t_input_pl: np.asarray(srnn_poses_t), 
        dec_out_pl: np.asarray(dec_true), is_training: True})


      if c_step % FLAGS.test_every == 0:
        encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch_srnn( test_set, actions[0] )
        _, srnn_poses_s, dec_true, _, _       = model.step_test(False, sess_1, encoder_inputs, decoder_inputs, decoder_outputs, sampling_rate=0.0)
        _, srnn_poses_t, _                    = model_t.step_test(False, sess_2, encoder_inputs, decoder_inputs, decoder_outputs, sampling_rate=0.0)

        loss_value_s, loss_value_t, loss_value_m, loss_value_i, srnn_poses_m, srnn_poses_i, w_s, w_t = sess.run(
                        [loss_s, loss_t, loss_m, loss_i, output_m_i, output_i, w_mapping_s, w_mapping_t],         # fake code here
                        feed_dict={s_input_pl: np.asarray(srnn_poses_s), 
                        t_input_pl: np.asarray(srnn_poses_t), 
                        dec_out_pl: np.asarray(dec_true), 
                        is_training: False})

        print ("loss mapping is {0:4f}, loss_seperate is {1:4f}, loss_s and loss_t are {2:4f}, {3:4f}".format(loss_value_m, 
          loss_value_i, loss_value_s, loss_value_t))
        print ("weights [s, t] are: {0:4f}, {1:4f}".format(w_s, w_t))

        if c_step % 50 == 0:
          print ("current c_step is: {0}".format(c_step))

        # if (loss_value_i < np.minimum(loss_value_s, loss_value_t)-0.002) or (loss_value_m < np.minimum(loss_value_s, loss_value_t)-0.002):
        # if (loss_value_i < np.minimum(loss_value_s, loss_value_t)+0.001) or (loss_value_m < np.minimum(loss_value_s, loss_value_t)+0.001):
        if c_step != 99:
        # if c_step % 20 == 0:
          srnn_gts_euler = get_srnn_gts( actions, model, test_set, data_mean, data_std, dim_to_ignore)
          srnn_gts_expmap = get_srnn_gts( actions, model, test_set, data_mean, data_std, dim_to_ignore, to_euler=False )

          mean_mean_errors_s, mean_mean_errors_t, mean_mean_errors_i, mean_mean_errors_m = angles_error(srnn_poses_s, 
            srnn_poses_t, srnn_poses_i, 
            srnn_poses_m, data_mean, 
            data_std, dim_to_ignore, actions, 
            number, srnn_gts_euler, srnn_gts_expmap)
    
          print()
          print("{0: <16} |".format("milliseconds"), end="")
          for ms in [80, 160, 240, 320, 400, 480, 560, 640, 720, 800, 880, 960, 1000]:
            print(" {0:5d} |".format(ms), end="")
          print()

          print("{0: <9}spatial |".format(action), end="")
          for ms in [1,3,5,7,9,11,13,15,17,19,21,23,24]:
            if FLAGS.seq_length_out >= ms+1:
              print(" {0:.3f} |".format( mean_mean_errors_s[ms] ), end="")
            else:
              print("   n/a |", end="")
          print()

          print("{0: <8}temporal |".format(action), end="")
          for ms in [1,3,5,7,9,11,13,15,17,19,21,23,24]:
            if FLAGS.seq_length_out >= ms+1:
              print(" {0:.3f} |".format( mean_mean_errors_t[ms] ), end="")
            else:
              print("   n/a |", end="")
          print()  

          print("{0: <8}weighted |".format(action), end="")
          for ms in [1,3,5,7,9,11,13,15,17,19,21,23,24]:
            if FLAGS.seq_length_out >= ms+1:
              print(" {0:.3f} |".format( mean_mean_errors_i[ms] ), end="")
            else:
              print("   n/a |", end="")
          print()  

          print("{0: <9}mapping |".format(action), end="")
          for ms in [1,3,5,7,9,11,13,15,17,19,21,23,24]:
            if FLAGS.seq_length_out >= ms+1:
              print(" {0:.3f} |".format( mean_mean_errors_m[ms] ), end="")
            else:
              print("   n/a |", end="")
          print()                      

          sampled_error_s   = np.mean(mean_mean_errors_s)
          sampled_error_t   = np.mean(mean_mean_errors_t) 
          sampled_error_m   = np.mean(mean_mean_errors_m) 
          sampled_error_i   = np.mean(mean_mean_errors_i)               

          print("spatial output mean error is:      ", sampled_error_s)
          print("temporal output mean error is:     ", sampled_error_t)
          print("mapping output mean error is:      ", sampled_error_m) 
          print("weighed output mean error is:      ", sampled_error_i)  


"""GX: autoencoder on 540 seperately"""

def main(_):
    train()
    # print("over")

if __name__ == "__main__":
  tf.app.run()

