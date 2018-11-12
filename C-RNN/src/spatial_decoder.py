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
        print ("decoder is of ", self.layers, " layers.")
        
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
 