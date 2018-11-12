
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

# Learning
tf.app.flags.DEFINE_float("learning_rate", 5e-5, "Learning rate.")    # maybe you could have a lower learning rate
tf.app.flags.DEFINE_float("sampling_rate", 1.0, "Learning rate.")
tf.app.flags.DEFINE_float("ending_sampling_rate", 0.2, "Learning rate.")
tf.app.flags.DEFINE_float("alpha", 1.0, "parameters before loss_1, loss_converge")
tf.app.flags.DEFINE_float("beta", 0.1, "parameters before loss_2, loss_reality")
tf.app.flags.DEFINE_float("gamma", 0.0, "parameters before loss_3, loss_balanced")
tf.app.flags.DEFINE_float("n_r", 0.0, "adding noise into training data")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95, "Learning rate is multiplied by this much. 1 means no decay.")
tf.app.flags.DEFINE_integer("learning_rate_step", 10000, "Every this many steps, do decay.")    # 10000
tf.app.flags.DEFINE_float("max_gradient_norm", 5, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("iterations", int(3e4), "Iterations to train for.")   # int(5e4)  2
tf.app.flags.DEFINE_integer("data_aug", 2, "data au")

# Architecture
tf.app.flags.DEFINE_string("architecture", "tied", "Seq2seq architecture to use: [basic, tied, attention].")
tf.app.flags.DEFINE_string("opt", "SGD", "optimizer to use: [SGD, Adam].")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer, this should be half of dimensions")
tf.app.flags.DEFINE_integer("d_layers", 2, "Spatial decoder's layer number, [0, 1, 2, 3], 0 means convolutional layers")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("seq_length_in", 50, "Number of frames to feed into the encoder. 25 fps")
tf.app.flags.DEFINE_integer("seq_length_out", 25, "Number of frames that the decoder has to predict.")
tf.app.flags.DEFINE_integer("long_t", 256, "Dimension number of long term information: [32, 64, 128, 256]")
tf.app.flags.DEFINE_boolean("residual_velocities", False, "Add a residual connection that effectively models velocities")

# Directories
tf.app.flags.DEFINE_string("data_dir", os.path.normpath("./data/h3.6m/dataset"), "Data directory")
tf.app.flags.DEFINE_string("train_dir", os.path.normpath("./experiments/"), "Training directory.")
tf.app.flags.DEFINE_boolean("CMU", False, "Would CMU mocap be used.")
tf.app.flags.DEFINE_string("action","all", "The action to train on. all means all the actions, all_periodic means walking, eating and smoking")
tf.app.flags.DEFINE_string("loss_to_use","sampling_based", "The type of loss to use, [sampling_based, conditioned_LSTM, scheduled_sampling, skeleton_sampling]")
tf.app.flags.DEFINE_string("x_s","inp_s", "The input for t: [inp_s, out_s]")

# remember to change test_every for quicker running time, the original value is 1000
tf.app.flags.DEFINE_integer("test_every", 500, "How often to compute error on the test set.")      # 1000
tf.app.flags.DEFINE_integer("save_every", 40000, "How often to compute error on the test set.")     # 25000
tf.app.flags.DEFINE_integer("round", 1, "round number, need to change it manually")
tf.app.flags.DEFINE_boolean("sample", False, "Set to True for sampling.")
tf.app.flags.DEFINE_boolean("use_cpu", False, "Whether to use the CPU")
tf.app.flags.DEFINE_integer("load", 0, "Try to load a previous checkpoint.")

FLAGS = tf.app.flags.FLAGS

if FLAGS.loss_to_use != "scheduled_sampling" and FLAGS.loss_to_use != "skeleton_sampling": 
  train_dir = os.path.normpath(os.path.join( FLAGS.train_dir, FLAGS.action,
    'out_{0}'.format(FLAGS.seq_length_out), 'data_aug_{0}'.format(FLAGS.data_aug),
    'iterations_{0}'.format(FLAGS.iterations),
    FLAGS.architecture,
    FLAGS.loss_to_use,
    'depth_{0}_size_{1}'.format(FLAGS.num_layers, FLAGS.size),
    'lr_{0}_sampling_rate_{1}'.format(FLAGS.learning_rate, FLAGS.sampling_rate),
    'residual_vel' if FLAGS.residual_velocities else 'not_residual_vel',
    'round_{0}'.format(FLAGS.round)))
else:
    train_dir = os.path.normpath(os.path.join( FLAGS.train_dir, FLAGS.action,
    'out_{0}'.format(FLAGS.seq_length_out), 'data_aug_{0}'.format(FLAGS.data_aug),
    'iterations_{0}'.format(FLAGS.iterations),
    FLAGS.architecture,
    FLAGS.loss_to_use,
    'starting_rate_{0}'.format(FLAGS.sampling_rate),
    'ending_sampling_rate_{0}'.format(FLAGS.ending_sampling_rate),
    'depth_{0}_size_{1}'.format(FLAGS.num_layers, FLAGS.size),
    'lr_{0}'.format(FLAGS.learning_rate),
    'residual_vel' if FLAGS.residual_velocities else 'not_residual_vel',
    'round_{0}'.format(FLAGS.round)))

summaries_dir = os.path.normpath(os.path.join( train_dir, "log" ))  # Directory for TB summaries

print ('framework\'s architecture is: ', FLAGS.architecture)

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
      summaries_dir,
      FLAGS.loss_to_use,
      optimizer_to_use,
      len( actions ),
      FLAGS.CMU,
      FLAGS.long_t,
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
  number_of_actions = len( actions )

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
    print( 'Noise rate is: ', FLAGS.n_r)
    print( "Model created" )

    srnn_gts_euler = get_srnn_gts( actions, model, test_set, data_mean, data_std, dim_to_ignore)

    #=== This is the training loop ===
    step_time, loss, val_loss = 0.0, 0.0, 0.0
    current_step = 0 if FLAGS.load <= 0 else FLAGS.load + 1
    previous_losses = []

    step_time, loss = 0, 0

    sampling_rate = FLAGS.sampling_rate


    sampling_rate_decay_factor = (FLAGS.sampling_rate - FLAGS.ending_sampling_rate)/ FLAGS.iterations

    for c_step in xrange( FLAGS.iterations ):

      start_time = time.time()

      # === Training step === Every time, you get a new batch and run your model on it to calculate the loss
      encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch( train_set, FLAGS.n_r )

      _, _, step_loss = model.step_train(True, sess, encoder_inputs, decoder_inputs, decoder_outputs, sampling_rate=sampling_rate)

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
        encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch_srnn( test_set, action, noise_rate = FLAGS.n_r )
        srnn_loss, srnn_poses, _ = model.step_test(False, sess, encoder_inputs, decoder_inputs, decoder_outputs, sampling_rate=0.0)

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
        print("{0: <13}RNN |".format(action), end="")
        for ms in [1,3,5,7,9,11,13,15,17,19,21,23,24]:
          if FLAGS.seq_length_out >= ms+1:
            print(" {0:.3f} |".format( mean_mean_errors[ms] ), end="")
          else:
            print("   n/a |", end="")
        print()

        sampled_error     = np.mean(mean_mean_errors)            

        print("spatial error is:         ", sampled_error)
        print("============================\n"
              "Global step:           %d\n"
              "Learning rate:       %.10f\n"
              "Step-time (ms):     %.4f\n"
              "Train loss avg:      %.4f\n"
              "sampling_rate:       %.4f\n"
              "--------------------------\n"
              "val loss:            %.4f\n"
              "srnn loss:           %.4f\n"
              "============================" % (model.global_step.eval(),
              model.learning_rate.eval(), step_time*1000, loss, sampling_rate_recoder, step_loss, srnn_loss))
        print()

        train_average_list.append(loss)
        mean_error_list.append(sampled_error)

        # Save the model
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
  plt.title('sequence in 50, out '+str(FLAGS.seq_length_out))
  plot_name = FLAGS.architecture+"_result_"+FLAGS.loss_to_use+"_"+FLAGS.action+"_alpha_"+str(FLAGS.alpha)+"_beta_"+str(FLAGS.beta)+"_gamma_"+str(FLAGS.gamma)+".png"
  plt.savefig(plot_name)

  print ('over')

def get_srnn_gts( actions, model, test_set, data_mean, data_std, dim_to_ignore, to_euler=True ):
  srnn_gts_euler = {}
  # print ("entering here")
  for action in actions:

    srnn_gt_euler = []
    _, _, srnn_expmap = model.get_batch_srnn( test_set, action, noise_rate = FLAGS.n_r )

    # expmap -> rotmat -> euler
    for i in np.arange( srnn_expmap.shape[0] ):
      denormed = data_utils.unNormalizeData(srnn_expmap[i,:,:], data_mean, data_std, dim_to_ignore, actions)

      if to_euler:
        for j in np.arange( denormed.shape[0] ):
          # print (denormed.shape)
          for k in np.arange(3,97,3):
            denormed[j,k:k+3] = data_utils.rotmat2euler( data_utils.expmap2rotmat( denormed[j,k:k+3] ))

      srnn_gt_euler.append( denormed );

    # Put back in the dictionary, every action will have 8 sequences of euler space
    srnn_gts_euler[action] = srnn_gt_euler
  # print (np.array(srnn_gts_euler[action]).shape)
  return srnn_gts_euler

def define_actions( action ):
  """
  Define the list of actions we are using.

  Args
    action: String with the passed action. Could be "all"
  Returns
    actions: List of strings of actions
  Raises
    ValueError if the action is not included in H3.6M
  """

  actions = ["walking", "eating", "smoking", "discussion",  "directions",
              "greeting", "phoning", "posing", "purchases", "sitting",
              "sittingdown", "takingphoto", "waiting", "walkingdog",
              "walkingtogether", "cmu"]

  if action in actions:
    return [action]

  if action == "all":
    return actions

  if action == "all_srnn":
    return ["walking", "eating", "smoking", "discussion"]

  raise( ValueError, "Unrecognized action: %d" % action )

def read_all_data( actions, seq_length_in, seq_length_out, data_dir):
  """
  Loads data for training/testing and normalizes it.

  Args
    actions: list of strings (actions) to load
    seq_length_in: number of frames to use in the burn-in sequence
    seq_length_out: number of frames to use in the output sequence
    data_dir: directory to load the data from
    one_hot: whether to use one-hot encoding per action
  Returns
    train_set: dictionary with normalized training data
    test_set: dictionary with test data
    data_mean: d-long vector with the mean of the training data
    data_std: d-long vector with the standard dev of the training data
    dim_to_ignore: dimensions that are not used becaused stdev is too small
    dim_to_use: dimensions that we are actually using in the model
  """

  # === Read training data ===
  print ("Reading training data (seq_len_in: {0}, seq_len_out {1}).".format(seq_length_in, seq_length_out))

  train_subject_ids = [1,6,7,8,9,11]     # for understanding purpose, train_subject_ids = [1,6,7,8,9,11]
  test_subject_ids = [5]

  train_set, complete_train = data_utils.load_data( data_dir, train_subject_ids, actions )
  test_set,  complete_test  = data_utils.load_data( data_dir, test_subject_ids,  actions )

  # Compute normalization stats
  data_mean, data_std, dim_to_ignore, dim_to_use = data_utils.normalization_stats(complete_train)

  # Normalize -- subtract mean, divide by stdev
  train_set = data_utils.normalize_data( train_set, data_mean, data_std, dim_to_use, actions )
  test_set  = data_utils.normalize_data( test_set,  data_mean, data_std, dim_to_use, actions )
  print("done reading data.")

  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use

def main(_):
  if FLAGS.sample:
    sample()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
