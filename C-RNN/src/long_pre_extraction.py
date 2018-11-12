
"""Simple code for training an RNN for motion prediction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import h5py

import numpy as np
from six.moves import xrange    # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import seq2seq_model
import matplotlib.pyplot as plt
from data_helper import define_actions, get_srnn_gts, read_all_data

# Learning
tf.app.flags.DEFINE_float("learning_rate", 5e-5, "Learning rate.")    # maybe you could have a lower learning rate
tf.app.flags.DEFINE_float("sampling_rate", 1.0, "Learning rate.")
tf.app.flags.DEFINE_float("ending_sampling_rate", 0.2, "Learning rate.")
tf.app.flags.DEFINE_float("alpha", 1.0, "parameters before loss_1, loss_converge")
tf.app.flags.DEFINE_float("beta", 0.2, "parameters before loss_2, loss_reality")
tf.app.flags.DEFINE_float("gamma", 0.0, "parameters before loss_3, loss_balanced")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95, "Learning rate is multiplied by this much. 1 means no decay.")
tf.app.flags.DEFINE_integer("learning_rate_step", 10000, "Every this many steps, do decay.")    # 10000
tf.app.flags.DEFINE_float("max_gradient_norm", 5, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("iterations", 15000, "Iterations to train for.")   # int(5e4)  2
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
tf.app.flags.DEFINE_integer("save_every", 1000, "How often to compute error on the test set.")     # 25000
tf.app.flags.DEFINE_integer("round", 1, "round number, need to change it manually")
tf.app.flags.DEFINE_boolean("sample", False, "Set to True for sampling.")
tf.app.flags.DEFINE_boolean("use_cpu", False, "Whether to use the CPU")
tf.app.flags.DEFINE_integer("load", 0, "Try to load a previous checkpoint.")

FLAGS = tf.app.flags.FLAGS

train_dir = os.path.normpath("temporal_model"+FLAGS.action)

def create_model(session, actions, optimizer_to_use, sampling=False):   # GX: here is to call the seq2seq model
  """Create translation model and initialize or load parameters in session."""

  model = seq2seq_model.Seq2SeqModel(
      FLAGS.architecture,
      FLAGS.seq_length_in if not sampling else 50,
      FLAGS.seq_length_out if not sampling else 600,  # 200,  # if sample, seq is 100
      FLAGS.size,                 # hidden layer size
      FLAGS.num_layers,
      FLAGS.max_gradient_norm,
      FLAGS.batch_size,
      FLAGS.learning_rate,
      FLAGS.learning_rate_decay_factor,
      FLAGS.loss_to_use,
      optimizer_to_use,
      len( actions ),
      FLAGS.CMU,
      FLAGS.alpha,
      FLAGS.beta,
      FLAGS.gamma,
      FLAGS.x_s,
      FLAGS.residual_velocities,
      FLAGS.d_layers,
      dtype=tf.float32)

  if FLAGS.load <= 0:
    print("Creating model with fresh parameters.")
    session.run(tf.global_variables_initializer())
    return model

  ckpt = tf.train.get_checkpoint_state( train_dir, latest_filename="checkpoint")
  print( "train_dir", train_dir )

  if ckpt and ckpt.model_checkpoint_path:
    # Check if the specific checkpoint exists
    if FLAGS.load > 0:
      if os.path.isfile(os.path.join(train_dir,"checkpoint-{0}.index".format(FLAGS.load))):
        ckpt_name = os.path.normpath(os.path.join( os.path.join(train_dir,"checkpoint-{0}".format(FLAGS.load)) ))
      else:
        raise ValueError("Asked to load checkpoint {0}, but it does not seem to exist".format(FLAGS.load))
    else:
      ckpt_name = os.path.basename( ckpt.model_checkpoint_path )

    print("Loading model {0}".format( ckpt_name ))
    model.saver.restore( session, ckpt.model_checkpoint_path )
    return model
  else:
    print("Could not find checkpoint. Aborting.")
    raise( ValueError, "Checkpoint {0} does not seem to exist".format( ckpt.model_checkpoint_path ) )

  return model

def train():
  """Train a seq2seq model on human motion"""

  actions = define_actions( FLAGS.action )    # here is like a list of actions, eg, ['walking'], ['walking, waiting, directions']

  train_average_list = []
  mean_error_list = []

  train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = read_all_data( actions, FLAGS.seq_length_in, FLAGS.seq_length_out, FLAGS.data_dir)
  number = 97

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)        # Allowing GPU memory growth, this means whole GPU
  device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}

  with tf.Session(config=tf.ConfigProto( gpu_options=gpu_options, device_count = device_count )) as sess:    

    # === Create the model ===
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))       # what is layer number in original paper?

    # print (FLAGS.opt)
    model = create_model( sess, actions, optimizer_to_use = FLAGS.opt )
    print( "Model created" )

    srnn_gts_euler = get_srnn_gts( actions, model, test_set, data_mean, data_std, dim_to_ignore)

    #=== This is the training loop ===
    step_time, loss, val_loss = 0.0, 0.0, 0.0
    current_step = 0 if FLAGS.load <= 0 else FLAGS.load
    previous_losses = []

    step_time, loss = 0, 0

    sampling_rate = FLAGS.sampling_rate


    sampling_rate_decay_factor = (FLAGS.sampling_rate - FLAGS.ending_sampling_rate)/ FLAGS.iterations

    for c_step in xrange( FLAGS.iterations ):

      start_time = time.time()

      # === Training step === Every time, you get a new batch and run your model on it to calculate the loss
      encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch( train_set )
      _, _, step_loss = model.step_train_t(True, sess, encoder_inputs, decoder_inputs, decoder_outputs, sampling_rate=sampling_rate)

      sampling_rate_recoder = sampling_rate
      sampling_rate = sampling_rate - sampling_rate_decay_factor
      step_time += (time.time() - start_time) / FLAGS.test_every
      loss += step_loss / FLAGS.test_every
      current_step += 1 

      # === step decay ===
      if current_step % FLAGS.learning_rate_step == 0:
        sess.run(model.learning_rate_decay_op)
        print ("the learning rate becomes to: {0}".format(model.learning_rate.eval()))

      # Once in a while, we save checkpoint, print statistics, and run evals. How to get more 160, 320ms's statistics?
      if current_step % FLAGS.test_every == 0:

        # === Validation with randomly chosen seeds ===     # only test it on subject 5's same action to get accuracy
        forward_only = True     # don't learn on these samples

        print()
        print("{0: <16} |".format("milliseconds"), end="")
        for ms in [80, 160, 240, 320, 400, 480, 560, 640, 720, 800, 880, 960, 1000]:
          print(" {0:5d} |".format(ms), end="")
        print()

        # # === Validation with srnn's seeds ===    # The only difference is how to choose the seed?
        action = actions[0]
        encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch_srnn( test_set, action )
        srnn_loss, srnn_poses, srnn_poses_s, srnn_poses_t, _ = model.step_test(False, sess, encoder_inputs, decoder_inputs, 
          decoder_outputs, sampling_rate=0.0)

        srnn_pred_expmap = data_utils.revert_output_format( srnn_poses, data_mean, data_std, dim_to_ignore, actions )

        # Save the errors here
        mean_errors = np.zeros( (len(srnn_pred_expmap), srnn_pred_expmap[0].shape[0]) )

        N_SEQUENCE_TEST = 8

        for i in np.arange(N_SEQUENCE_TEST):
          eulerchannels_pred = srnn_pred_expmap[i]         

          # Convert from exponential map to Euler angles
          for j in np.arange( eulerchannels_pred.shape[0] ):
            for k in np.arange(3,number,3):
              eulerchannels_pred[j,k:k+3] = data_utils.rotmat2euler(
                data_utils.expmap2rotmat( eulerchannels_pred[j,k:k+3] ))                           

          gt_i=np.copy(srnn_gts_euler[action][i])
          gt_i[:,0:6] = 0     # the translation is 0?

          idx_to_use  = np.where( np.std( gt_i, 0 ) > 1e-4 )[0]                    
          euc_error   = np.power( gt_i[:,idx_to_use] - eulerchannels_pred[:,idx_to_use], 2)         
          euc_error   = np.sum(euc_error, 1)
          euc_error   = np.sqrt( euc_error )        
          mean_errors[i,:]    = euc_error

        # This is simply the mean error over the N_SEQUENCE_TEST examples
        mean_mean_errors   = np.mean( mean_errors, 0 )

        # GX: here I think 1000ms for 25 frames, so that you calculate error on one single frame
        print("{0: <8}temporal |".format(action), end="")
        for ms in [1,3,5,7,9,11,13,15,17,19,21,23,24]:
          if FLAGS.seq_length_out >= ms+1:
            print(" {0:.3f} |".format( mean_mean_errors[ms] ), end="")
          else:
            print("   n/a |", end="")
        print()

        sampled_error     = np.mean(mean_mean_errors)              

        print("temporal output mean error is:     ", sampled_error)
        print("============================\n"
              "Global step:           %d\n"
              "Learning rate:       %.10f\n"
              "Step-time (ms):     %.4f\n"
              "Train loss avg:      %.4f\n"
              "sampling_rate:       %.4f\n"
              "--------------------------\n"
              "val loss:            %.4f\n"
              "srnn loss:           %.4f\n"
              "============================" % (current_step,
              model.learning_rate.eval(), step_time*1000, loss, sampling_rate_recoder, step_loss, srnn_loss))
        print()

        train_average_list.append(loss)
        mean_error_list.append(sampled_error)

        # Save the model
        print ("current step is : ", current_step)
        if current_step % FLAGS.save_every == 0:
          print( "Saving the model..." ); start_time = time.time()
          model.saver.save(sess, os.path.normpath(os.path.join(train_dir, 'checkpoint')), global_step=current_step )
          print( "done in {0:.2f} ms".format( (time.time() - start_time)*1000) )

        # Reset global time and loss
        step_time, loss = 0, 0
        sys.stdout.flush()

  fig, ax = plt.subplots()
  ax.plot(train_average_list, 'r-', label='train loss')
  ax.plot(mean_error_list, 'b-', label='mean error')
  legend = ax.legend(loc=0)
  plt.grid(True)
  plt.title('sequence in '+str(FLAGS.seq_length_in)+', out '+str(FLAGS.seq_length_out))
  plot_name = FLAGS.action+"_alpha_"+str(FLAGS.alpha)+"_beta_"+str(FLAGS.beta)+"_gamma_"+str(FLAGS.gamma)+".png"
  plt.savefig(plot_name)

  print ('over')

def sample():

  actions = define_actions( FLAGS.action )    # here is like a list of actions, eg, ['walking'], ['walking, waiting, directions']

  _, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = read_all_data( actions, FLAGS.seq_length_in, FLAGS.seq_length_out, FLAGS.data_dir)
  number = 97

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)        # Allowing GPU memory growth, this means whole GPU
  device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}

  with tf.Session(config=tf.ConfigProto( gpu_options=gpu_options, device_count = device_count )) as sess:    

    model = create_model( sess, actions, optimizer_to_use = FLAGS.opt )
    print( "Model created" )

    srnn_gts_euler = get_srnn_gts( actions, model, test_set, data_mean, data_std, dim_to_ignore)

    print()
    print("{0: <16} |".format("milliseconds"), end="")
    for ms in [80, 160, 240, 320, 400, 480, 560, 640, 720, 800, 880, 960, 1000]:
      print(" {0:5d} |".format(ms), end="")
    print()

    # # === Validation with srnn's seeds ===    # The only difference is how to choose the seed?
    action = actions[0]
    encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch_srnn( test_set, action )
    srnn_loss, srnn_poses, srnn_poses_s, srnn_poses_t, _ = model.step_test(False, sess, encoder_inputs, decoder_inputs, 
      decoder_outputs, sampling_rate=0.0)

    srnn_pred_expmap = data_utils.revert_output_format( srnn_poses, data_mean, data_std, dim_to_ignore, actions )

    # Save the errors here
    mean_errors = np.zeros( (len(srnn_pred_expmap), srnn_pred_expmap[0].shape[0]) )

    N_SEQUENCE_TEST = 8

    for i in np.arange(N_SEQUENCE_TEST):
      eulerchannels_pred = srnn_pred_expmap[i]         

      # Convert from exponential map to Euler angles
      for j in np.arange( eulerchannels_pred.shape[0] ):
        for k in np.arange(3,number,3):
          eulerchannels_pred[j,k:k+3] = data_utils.rotmat2euler(
            data_utils.expmap2rotmat( eulerchannels_pred[j,k:k+3] ))                           

      gt_i=np.copy(srnn_gts_euler[action][i])
      gt_i[:,0:6] = 0     # the translation is 0?

      idx_to_use  = np.where( np.std( gt_i, 0 ) > 1e-4 )[0]                    
      euc_error   = np.power( gt_i[:,idx_to_use] - eulerchannels_pred[:,idx_to_use], 2)         
      euc_error   = np.sum(euc_error, 1)
      euc_error   = np.sqrt( euc_error )        
      mean_errors[i,:]    = euc_error

    # This is simply the mean error over the N_SEQUENCE_TEST examples
    mean_mean_errors   = np.mean( mean_errors, 0 )

    # GX: here I think 1000ms for 25 frames, so that you calculate error on one single frame
    print("{0: <8}temporal |".format(action), end="")
    for ms in [1,3,5,7,9,11,13,15,17,19,21,23,24]:
      if FLAGS.seq_length_out >= ms+1:
        print(" {0:.3f} |".format( mean_mean_errors[ms] ), end="")
      else:
        print("   n/a |", end="")
    print()

    sampled_error     = np.mean(mean_mean_errors)              

    print("temporal output mean error is:     ", sampled_error)
    print()


def main(_):
  if FLAGS.sample:
    sample()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
