import tensorflow as tf
import tensorflow.contrib.layers as tcl

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

class AEDecoder(object):
    def __init__(self, re_term, layers, output_dim, name_scope='DecoderSkeleton'):
        self.re_term=re_term
        self.layers = layers
        self.name_scope = name_scope
        self.output_dim = output_dim
        # print ("decoder is of ", self.layers, " layers.")
        
    def forward(self, single_input, is_training):
        with tf.variable_scope(self.name_scope) as vs:

            h0 = tcl.fully_connected(single_input, 512, scope="fc3", activation_fn=lrelu, 
              weights_regularizer=tcl.l2_regularizer(self.re_term))

            h0 = tf.where(is_training, tcl.dropout(h0, keep_prob = 0.5, is_training=True), h0)
            # GX: maybe you can check out overfitting later...
            h0 = tcl.fully_connected(h0, self.output_dim, scope="fc4", activation_fn=None, 
              weights_regularizer=tcl.l2_regularizer(self.re_term),)

            h0 = tf.expand_dims(tf.expand_dims(h0, 1), 3)
            h0 = tf.reshape(h0, [-1, h0.get_shape()[2].value])

            return h0

class convEncoder(object):
  def __init__(self, output_dim, re_term, channels=64, name_scope = "EncoderSkeleton"):
    self.output_dim = output_dim
    self.channels = channels
    self.re_term = re_term
    self.name_scope = name_scope

  def forward(self, skeleton_input, is_training):

    with tf.variable_scope(self.name_scope) as vs:

      h0 = lrelu(tcl.conv2d(tf.reshape(skeleton_input, [-1, 1, skeleton_input.get_shape()[1].value, 1]),
        num_outputs=self.channels,
        stride=1, kernel_size=[1, 3],
        activation_fn=None, padding='SAME', scope="conv1",
        weights_regularizer=tcl.l2_regularizer(self.re_term), biases_initializer=None))
      h0 = tf.where(is_training, tcl.dropout(h0, 0.8, is_training=True), h0)


      h0 = lrelu(tcl.conv2d(h0,
        num_outputs=self.channels,
        stride=1, kernel_size=[1, 3],
        activation_fn=None, padding='SAME', scope="conv2",
        weights_regularizer=tcl.l2_regularizer(self.re_term), biases_initializer=None))
      h0 = tf.where(is_training, tcl.dropout(h0, 0.8, is_training=True), h0)

      h0 = tcl.flatten(h0)
      h0 = tcl.fully_connected(h0, self.output_dim, weights_regularizer=tcl.l2_regularizer(self.re_term), 
        scope="fc1", activation_fn=lrelu)

      return h0  
      