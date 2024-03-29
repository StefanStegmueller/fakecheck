#!/usr/bin/python

import datetime
from transformers import TFBertForSequenceClassification
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import backend as K, initializers, regularizers, constraints
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Layer, Dropout, LSTM, Dense, InputLayer
from tensorflow.keras.losses import Loss


class Attention(Layer):
    """
    SOURCE: https://gist.github.com/cbaziotis/6428df359af27d58078ca5ed9792bd6d
    """
    
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True,
                 return_attention=False,
                 **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Note: The layer has been tested with Keras 1.x
        Example:
        
            # 1
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...
            # 2 - Get the attention scores
            hidden = LSTM(64, return_sequences=True)(words)
            sentence, word_scores = Attention(return_attention=True)(hidden)
        """
        self.supports_masking = True
        self.return_attention = return_attention
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'supports_masking': self.supports_masking,
            'return_attention': self.return_attention,
            'init': self.init,
            'W_regularizer': self.W_regularizer,
            'b_regularizer': self.b_regularizer,
            'W_constraint': self.W_constraint,
            'b_constraint': self.b_constraint,
            'bias': self.bias,
        })
        return config

    def build(self, input_shape):
        assert len(input_shape) == 3

        
        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        eij = dot_product(x, self.W)

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        weighted_input = x * K.expand_dims(a)

        result = K.sum(weighted_input, axis=1)

        if self.return_attention:
            return [result, a]
        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]),
                    (input_shape[0], input_shape[1])]
        else:
            return input_shape[0], input_shape[-1]
        
        
class RankingError(Loss):   
    
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
    
    def call(self, y_true, y_diff):
        pos = tf.constant([1.0 for i in range(self.batch_size)])
        neg = tf.constant([-1.0 for i in range(self.batch_size)])
        sign = tf.where(tf.equal(y_true,1.0), pos, neg)

        return tf.math.maximum(0.0, 1.0 - sign * y_diff)

    
def dot_product(x, kernel):
    """
    SOURCE: https://gist.github.com/cbaziotis/6428df359af27d58078ca5ed9792bd6d
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        # todo: check that this is correct
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


def build_base_model(input_shape, hidden_units, dropout_prob, model_name='base'):
    model = Sequential(name=model_name)
    model.add(LSTM(hidden_units, input_shape=input_shape, return_sequences=True, name='lstm'))
    model.add(Attention(name='attention'))
    model.add(Dropout(dropout_prob))
    model.add(Dense(1, activation='sigmoid', name='dense'))
    return model


def build_ranking_model(base_forward_func, input1, input2):       
    out_s1 = base_forward_func(input1)
    out_s1 = Layer(name='out_s1')(out_s1)
    out_s2 = base_forward_func(input2) 
    out_diff = Layer(name='out_diff')(tf.math.subtract(out_s1, out_s2, name='out_diff'))
    
    if isinstance(input1, list) and isinstance(input2, list):
        total_inputs = input1 + input2
    else:
        total_inputs = [input1] + [input2]
        
    return tf.keras.Model(inputs=total_inputs, outputs=[out_s1, out_diff], name='ranking')


def load_bert_model(model_path):
    cbert_model = TFBertForSequenceClassification.from_pretrained(model_path)
    cbert_model.classifier.activation = tf.keras.activations.sigmoid
    return cbert_model