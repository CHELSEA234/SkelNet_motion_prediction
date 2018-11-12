import tensorflow as tf
import os

import seq2seq_model
import seq2seq_model_t

def create_model_s(session, actions, FLAGS, sampling=False):   # GX: here is to call the seq2seq model
  """Create translation model and initialize or load parameters in session."""
  train_dir_s = os.path.normpath("spatial_model_"+FLAGS.action)
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
      FLAGS.opt,
      len( actions ),
      FLAGS.CMU,
      FLAGS.alpha,
      FLAGS.beta,
      FLAGS.gamma,
      FLAGS.x_s,
      FLAGS.residual_velocities,
      FLAGS.d_layers,
      dtype=tf.float32)

  ckpt = tf.train.get_checkpoint_state( train_dir_s, latest_filename="checkpoint")

  if ckpt and ckpt.model_checkpoint_path:
    if FLAGS.load > 0:
      ckpt_name = os.path.normpath(os.path.join( os.path.join(train_dir_s,"checkpoint-{0}".format(FLAGS.load)) ))

    model.saver.restore( session, ckpt.model_checkpoint_path )
    return model

  return model

def create_model_t(session, actions, FLAGS, sampling=False):   # GX: here is to call the seq2seq model
  """Create translation model and initialize or load parameters in session."""
  train_dir_t = os.path.normpath("temporal_model_"+FLAGS.action)
  model = seq2seq_model_t.Seq2SeqModel(
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
      FLAGS.opt,
      len( actions ),
      FLAGS.CMU,
      FLAGS.alpha,
      FLAGS.beta,
      FLAGS.gamma,
      FLAGS.x_s,
      FLAGS.residual_velocities,
      FLAGS.d_layers,
      dtype=tf.float32)

  ckpt = tf.train.get_checkpoint_state( train_dir_t, latest_filename="checkpoint")

  if ckpt and ckpt.model_checkpoint_path:
    if FLAGS.load_t > 0:
      ckpt_name = os.path.normpath(os.path.join( os.path.join(train_dir_t,"checkpoint-{0}".format(FLAGS.load_t)) ))

    model.saver.restore( session, ckpt.model_checkpoint_path )
    return model

  return model  

