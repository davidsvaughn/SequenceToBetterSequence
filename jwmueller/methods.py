"""  Helper functions for VariationalModel class """

from __future__ import print_function
from __future__ import division
import tensorflow as tf
import numpy as np
import math
import random
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import *

def linearOutcomePrediction(zs, params_pred, use_sigmoid = False, scope = None):
    """ Model for predicting outcomes from latent representations Z.
    zs = batch of z-vectors (encoder-states, matrix).
    """
    with variable_scope.variable_scope(scope or "outcomepred", reuse = True):
        coefficients, bias = params_pred
        outcome_preds = tf.add(tf.matmul(zs, coefficients), bias)
    return outcome_preds

def flexibleOutcomePrediction(zs, params_pred, use_sigmoid = False, scope = None):
    """ Model for nonlinearly predicting outcomes from latent representations Z.
        Uses a single hidden layer of pre-specified size, by default = d (the size of the RNN hidden-state)
    zs = batch of z-vectors (encoder-states, matrix).
    use_sigmoid: if True, then outcome-predictions are constrained to [0,1]
    """
    with variable_scope.variable_scope(scope or "outcomepred", reuse = True):
        weights_pred = params_pred[0]
        biases_pred = params_pred[1]
        hidden1 = tf.nn.tanh(tf.add(tf.matmul(zs,weights_pred['W1']), biases_pred['B1']))
        outcome_preds = tf.add(tf.matmul(hidden1,weights_pred['W2']), biases_pred['B2'])
        if use_sigmoid:
            outcome_preds = tf.sigmoid(outcome_preds)
    return outcome_preds

def outcomePrediction(zs, params_pred, which_outcomeprediction, use_sigmoid = False, scope = None):
    if which_outcomeprediction == 'linear':
        return linearOutcomePrediction(zs, params_pred, scope = scope)
    else:
        return flexibleOutcomePrediction(zs, params_pred, scope = scope)

def getEncoding(inputs, cell, num_symbols, embedding_size, 
                dtype=dtypes.float32, scope=None):
    """ Model for produce encoding z from x.
    zs = batch of z-vectors (encoder-states, matrix).
    """
    with variable_scope.variable_scope(scope or "seq2seq", reuse = True):
        encoder_cell = core_rnn_cell.EmbeddingWrapper(
            cell, embedding_classes=num_symbols,
            embedding_size=embedding_size)
        _, encoder_state = rnn.static_rnn(encoder_cell, inputs, dtype=dtype) # batch_size x cell.state_size
    return(encoder_state)

def variationalEncoding(inputs, cell, num_symbols, embedding_size, 
                variational_params, dtype=dtypes.float32, scope=None):
    """ Model for produce encoding z from x.
    zs: batch of z-vectors (encoder-states, matrix).
    sigmas: posterior standard devs for each dimension, poduced using 2-layer neural net with Relu units.
    """
    min_sigma = 1e-6 # the smallest allowable sigma value.
    h_T = getEncoding(inputs, cell, num_symbols, embedding_size, 
                dtype=dtypes.float32, scope=scope)
    with variable_scope.variable_scope(scope or "variational", reuse = True):
        mu_params, sigma_params = variational_params
        mu = tf.add(tf.matmul(h_T,mu_params['weights']),mu_params['biases'])
        #sigma = tf.sigmoid(tf.add(tf.matmul(h_T,sigma_params['weights']), 
        #                          sigma_params['biases']))
        hidden_layer_sigma = tf.nn.relu(tf.add(tf.matmul(h_T,sigma_params['weights1']), 
                                  sigma_params['biases1'])) # Relu layer of same size as h_T
        sigma = tf.clip_by_value(tf.exp(-tf.abs(tf.add(tf.matmul(hidden_layer_sigma,sigma_params['weights2']), 
                                  sigma_params['biases2']))),min_sigma,1.0)
    return(mu, sigma)

def getDecoding(encoder_state, inputs, cell,
                num_symbols, embedding_size,
                feed_previous = True, output_projection=None,
                dtype=dtypes.float32, scope=None):
    """ Model for producing probabilities over x from z """
    with variable_scope.variable_scope(scope or "seq2seq", reuse = True):
        if output_projection is None:
            cell = core_rnn_cell.OutputProjectionWrapper(cell, num_symbols)
        decode_probs, _ = embedding_rnn_decoder(
             inputs, encoder_state, cell, num_symbols,
             embedding_size, output_projection=output_projection,
             feed_previous=feed_previous)
    return(decode_probs)

def createVariationalVar(inputs, cell, num_symbols, embedding_size, 
                    feed_previous = False, output_projection=None,
                    dtype=dtypes.float32,  scope=None):
    """ Creates Tensorflow variables which can be reused. """
    with variable_scope.variable_scope(scope or "seq2seq"):
        encoder_cell = core_rnn_cell.EmbeddingWrapper(
            cell, embedding_classes=num_symbols,
            embedding_size=embedding_size)
        _, encoder_state = rnn.static_rnn(encoder_cell, inputs, dtype=dtype) # batch_size x cell.state_size
        if output_projection is None:
            cell = core_rnn_cell.OutputProjectionWrapper(cell, num_symbols)
        decode_probs, _ = embedding_rnn_decoder(
             inputs, encoder_state, cell, num_symbols,
             embedding_size, output_projection=output_projection,
             feed_previous=feed_previous)
    return None

def createDeterministicVar(inputs, cell, num_symbols, embedding_size, 
                    feed_previous = False, output_projection=None,
                    dtype=dtypes.float32,  scope=None):
    """ Creates Tensorflow variables which can be reused. """
    with variable_scope.variable_scope(scope or "seq2seq"):
        encoder_cell = core_rnn_cell.EmbeddingWrapper(
            cell, embedding_classes=num_symbols,
            embedding_size=embedding_size)
        _, encoder_state = rnn.static_rnn(encoder_cell, inputs, dtype=dtype) # batch_size x cell.state_size
        if output_projection is None:
            cell = core_rnn_cell.OutputProjectionWrapper(cell, num_symbols)
        decode_probs, _ = embedding_rnn_decoder(
             inputs, encoder_state, cell, num_symbols,
             embedding_size, output_projection=output_projection,
             feed_previous=feed_previous)
    return None

""" Computes edit distance between two (possibly padded) sequences: """
def levenshtein(seq1, seq2):
    s1 = [value for value in seq1 if value != '<PAD>']
    s2 = [value for value in seq2 if value != '<PAD>']
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

""" Performs random edits of sequence, respecting min/max sequence-length constraints.
    At each edit, possible operations (equally likely) are:
    (1) Do nothing  (2) Substitution  (3) Deletion  (4) Insertion
    Each operation is uniform over possible symbols and possible positions.
"""
def mutate_lengthconstrained(init_seq, num_edits, vocab, length_range = (10,20)):
    min_seq_length, max_seq_length = length_range
    new_seq = init_seq[:]
    for i in range(num_edits):
        operation = random.randint(1,4) # 1 = Do nothing, 2 = Substitution, 3 = Deletion, 4 = Insertion
        if operation > 1:
            char = '<PAD>' # potential character, cannot be PAD.
            while char == '<PAD>':
                char = vocab[random.randint(0, len(vocab)-1)]
            position = random.randint(0, len(new_seq)-1)
            if (operation == 4) and (len(new_seq) < max_seq_length): # Insertion
                position = random.randint(0, len(new_seq))
                new_seq.insert(position,char)
            elif operation == 2: # Substitution
                new_seq[position] = char
            elif (operation == 3) and (len(new_seq) > min_seq_length): # Deletion
                _ = new_seq.pop(position)
    edit_dist = levenshtein(new_seq, init_seq)
    if edit_dist > num_edits:
        raise ValueError("edit distance invalid")
    return(new_seq, edit_dist)

""" Performs random edits of sequence, ignoring sequence-length constraints (except length must be > 0).
    At each edit, possible operations (equally likely) are:
    (1) Do nothing  (2) Substitution  (3) Deletion  (4) Insertion
    Each operation is uniform over possible symbols and possible positions.
"""
def mutate(init_seq, num_edits, vocab):
    new_seq = init_seq[:]
    for i in range(num_edits):
        operation = random.randint(1,4) # 1 = Do nothing, 2 = Substitution, 3 = Deletion, 4 = Insertion
        if operation > 1:
            char = '<PAD>' # potential character, cannot be PAD.
            while char == '<PAD>':
                char = vocab[random.randint(0, len(vocab)-1)]
            position = random.randint(0, len(new_seq)-1)
            if (operation == 4): # Insertion
                position = random.randint(0, len(new_seq))
                new_seq.insert(position,char)
            elif operation == 2: # Substitution
                new_seq[position] = char
            elif (operation == 3) and len(new_seq) > 1: # Deletion
                _ = new_seq.pop(position)
    edit_dist = levenshtein(new_seq, init_seq)
    if edit_dist > num_edits:
        raise ValueError("edit distance invalid")
    return(new_seq, edit_dist)

def sigmoid(x):
    return(1 / (1 + math.exp(-x)))

def smoothedsigmoid(x, b=1):
    """ b controls smoothness, lower = smoother """
    return(1 / (1 + math.exp(-b*x)))
