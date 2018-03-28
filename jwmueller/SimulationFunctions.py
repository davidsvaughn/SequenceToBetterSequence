""" Helper functions for generating sequences + outcomes from simulation grammar """

from __future__ import print_function
from __future__ import division
import numpy as np
import random

def generateSimulationData(num_sample = 1000, vocab_size = 10, length_range=(10,20)):
    min_seq_length,max_seq_length = length_range
    seqs = []
    true_outcomes = []
    for i in range(num_sample):
        seq = generateSimulationSeq(random.randint(min_seq_length,max_seq_length), vocab_size)
        seqs.append(seq)
        true_outcomes.append(getSimulationOutcome(seq, max_seq_length))
    return(seqs, true_outcomes)

def generateSimulationSeq(seq_length, vocab_size):
    alphabet = "ABCDEFGHIJKLMNOZQRSTUVWXYZ"
    seq = ['Z']
    ninetyfive_percent_const = 19.0 # use 19 for 95 percent
    fifty_percent_const = 1.0 # use 1 for 50 percent
    for t in range(1,seq_length+1): # draw next character
        probs = np.array([1.0]*vocab_size)
        if (t >= 1) and (seq[t-1] == 'A'):
            probs[alphabet.index('B')] = ninetyfive_percent_const*(vocab_size - 1) 
        if (t >= 3) and (seq[t-3] == 'D'):
            probs[alphabet.index('D')] = ninetyfive_percent_const*(vocab_size - 1) 
        if (t >= 5) and (seq[t-5] == 'E'):
            probs[alphabet.index('E')] = ninetyfive_percent_const*(vocab_size - 1) 
        if (t >= 2) and (seq[t-2] == 'H') and (seq[t-1] == 'I'):
            probs[alphabet.index('J')] = ninetyfive_percent_const*(vocab_size - 1) 
        if (t >= 2) and (seq[t-2] == 'I') and (seq[t-1] == 'H'):
            probs[alphabet.index('I')] = ninetyfive_percent_const*(vocab_size - 1) 
        if (t >= 3) and (seq[t-3] == 'B') and (seq[t-2] == 'C'):
            probs[alphabet.index('B')] = ninetyfive_percent_const*(vocab_size - 1)
        if (t >= 11) and (seq[t-1] == 'F'):
            probs[alphabet.index('F')] = ninetyfive_percent_const*(vocab_size - 1)
        if (t == 7) and (seq[t-1] == 'F'):
            probs[alphabet.index('G')] = ninetyfive_percent_const*(vocab_size - 1)
        if (t == 8) and (seq[t-1] == 'F'):
            probs[alphabet.index('G')] = fifty_percent_const*(vocab_size - 1) 
        if (t == 5) or (t == 10) or (t == 15) or (t == 20):
            probs[alphabet.index('C')] = fifty_percent_const*(vocab_size - 1)
        if (t >= 1) and (seq[t-1] == 'C'):
            probs[alphabet.index('A')] = fifty_percent_const*(vocab_size - 1)
        next_char = alphabet[np.random.choice(vocab_size, p= probs/np.sum(probs))]
        seq.append(next_char)
    return(seq[1:])

""" Counts number of A occurrences, rescaled by max_seq_length """
def getSimulationOutcome(seq, max_seq_length = 20.0):
    return(seq.count('A')/max_seq_length)

def simulationSeqProbability(char_seq, vocab_size=10,length_range=(10,20)):
    """ Returns log probability of a given sequence in underlying generative model """
    seq_length = len(char_seq)
    min_log_prob = -1e6 # smallest possible return value.
    if seq_length < length_range[0] or seq_length > length_range[1]:
        return(min_log_prob)
    seq = char_seq[:]
    seq.insert(0,'Z')
    alphabet = "ABCDEFGHIJKLMNOZQRSTUVWXYZ"
    ninetyfive_percent_const = 19.0 # use 19 for 95 percent
    fifty_percent_const = 1.0 # use 1 for 50 percent
    log_prob = 0.0
    for t in range(1,seq_length+1): # draw next character
        probs = np.array([1.0]*vocab_size)
        if (t >= 1) and (seq[t-1] == 'A'):
            probs[alphabet.index('B')] = ninetyfive_percent_const*(vocab_size - 1) 
        if (t >= 3) and (seq[t-3] == 'D'):
            probs[alphabet.index('D')] = ninetyfive_percent_const*(vocab_size - 1) 
        if (t >= 5) and (seq[t-5] == 'E'):
            probs[alphabet.index('E')] = ninetyfive_percent_const*(vocab_size - 1) 
        if (t >= 2) and (seq[t-2] == 'H') and (seq[t-1] == 'I'):
            probs[alphabet.index('J')] = ninetyfive_percent_const*(vocab_size - 1) 
        if (t >= 2) and (seq[t-2] == 'I') and (seq[t-1] == 'H'):
            probs[alphabet.index('I')] = ninetyfive_percent_const*(vocab_size - 1) 
        if (t >= 3) and (seq[t-3] == 'B') and (seq[t-2] == 'C'):
            probs[alphabet.index('B')] = ninetyfive_percent_const*(vocab_size - 1)
        if (t >= 11) and (seq[t-1] == 'F'):
            probs[alphabet.index('F')] = ninetyfive_percent_const*(vocab_size - 1)
        if (t == 7) and (seq[t-1] == 'F'):
            probs[alphabet.index('G')] = ninetyfive_percent_const*(vocab_size - 1)
        if (t == 8) and (seq[t-1] == 'F'):
            probs[alphabet.index('G')] = fifty_percent_const*(vocab_size - 1) 
        if (t == 5) or (t == 10) or (t == 15) or (t == 20):
            probs[alphabet.index('C')] = fifty_percent_const*(vocab_size - 1)
        if (t >= 1) and (seq[t-1] == 'C'):
            probs[alphabet.index('A')] = fifty_percent_const*(vocab_size - 1)
        next_char = seq[t]
        probs = probs/np.sum(probs)
        log_prob += np.log(probs[alphabet.index(next_char)])
    return(log_prob + np.log(1/(length_range[1]-length_range[0]+1)))

def restrictLength(revised_seq, length_range=(10,20)):
    """ Simply cuts of end of list or duplicates last symbol until len(list) is in length_range. """
    while len(revised_seq) > length_range[1]:
        del revised_seq[-1]
    while len(revised_seq) < length_range[0]:
        char = revised_seq[-1]
        revised_seq.append(char)
    return revised_seq

""" Converts list of ints to alphabetical chars """
def toAlphabet(int_list):
    alphabet = "ABCDEFGHIJKLMNOZQRSTUVWXYZ"
    char_list = [''] * len(int_list)
    for t in range(len(int_list)):
        char_list[t] = alphabet[int_list[t]]
    return(char_list)

