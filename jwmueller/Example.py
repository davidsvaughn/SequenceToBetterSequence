""" Toy Example to demonstrate basic usage of our methods: 
    Revising sequences generated from simple grammar.
    The associated outcome is the count of the 'A' symbol.  
"""

from __future__ import print_function
from __future__ import division
import numpy as np
import random
import VariationalModel
from SimulationFunctions import *
from scipy.stats.stats import pearsonr
import pickle

vocab_size = 10
vocab = set()
alphabet = "ABCDEFGHIJKLMNOZQRSTUVWXYZ"
for i in range(vocab_size):
   vocab.add(alphabet[i])

vocab = sorted(vocab) # Vocabulary should be list with unique entries

# Create datasets:
max_seq_length = 20 
length_range = (10,max_seq_length)
n_train = 10000
n_val = 1000
n_test = 1000

train_data = generateSimulationData(n_train,vocab_size,length_range)
val_data = generateSimulationData(n_val,vocab_size,length_range)
test_data = generateSimulationData(n_test,vocab_size,length_range)
print("training sequences:", train_data[0][:3]) # three example sequences.
print("training outcomes:", train_data[1][:3]) # three example outcomes.
print('std dev of outcomes:', np.std(train_data[1]))

# Setup seq2betterSeq model:
model = VariationalModel.VariationalModel(max_seq_length = max_seq_length, vocab = vocab, use_unknown = False, 
            learning_rate = 1e-3, embedding_dim = 8,  memory_dim = 128, use_sigmoid = True,
            outcome_var = np.var(np.array(train_data[1])), logdir = 'output/logs/')

# model.resetLearnRate(1e-6) # Change learning rate
# tensorboard --logdir=output/logs/ # Tensorboard training summary

# Train model:
train_losses, val_outcome_errors, val_seq2seq_errors = model.train(
        train_seqs=train_data[0], train_outcomes=train_data[1], 
        test_seqs = val_data[0], test_outcomes = val_data[1],
        which_model = 'joint', seq2seq_importanceval = 0.95,
        kl_importanceval = 0.0, invar_importanceval = 0.0,
        max_epochs = 50, invar_batchsize = 32)

# Save model:
# model.save('GRU_embed9_mem128_invar')

# Models:

# Evaluate Prediction Performance on test data:
predicted_outcomes, reconstructed_seqs, avg_outcome_error, avg_reconstruction_error, avg_sigma = model.predict(test_data[0],test_data[1])
print("Reconstruction err:", avg_reconstruction_error)
print("Predicted Outcome Error:", avg_outcome_error, "  (Test) R^2:",  1- avg_outcome_error/np.var(test_data[1]))
print("Correlation of predictions and outcomes:", pearsonr(predicted_outcomes, test_data[1])[0])
print("Avg Sigma:", avg_sigma)

# Save model:
# model.save('/Users/jonasmueller/Dropbox (Personal)/CombinatorialOpt/Code/output/models/simulationGRUembed8mem64')
# Restore model:
# model.restore('/Users/jonasmueller/Dropbox (Personal)/CombinatorialOpt/Code/output/models/simulationGRUembed8mem64')

""" In practice, need to train model as follows:

1)  First train with kl_importanceval = 0, invar_importanceval = 0
    until val_reconstruction_error + val_outcome_error plateau.

2)  Find a good setting of seq2seq_importanceval which
    leads to low val_reconstruction_error and val_outcome_error

3)  Carefully begin increasing kl_importanceval from 0 to 1 and continue training.
    Make sure to save your model every few training epochs.
    Anytime val_reconstruction/outcome_error suddenly becomes worse, need to: 
    - halt training and reload last model where val_reconstruction/outcome_error was still good
    - slightly lower kl_importanceval and resume training

4)  Carefully begin increasing invar_importanceval.
    Make sure to save your model every few training epochs.
    Anytime val_reconstruction/outcome_error suddenly becomes worse, need to: 
    - halt training and reload last model where val_reconstruction/outcome_error was still good
    - slightly lower invar_importanceval and resume training
    Keep increasing invar_importanceval until val_reconstruction/outcome_error always begins to worsen. 
"""

# Revise this single sequence:
init_seq, init_outcome = generateSimulationData(1)
log_alpha = -10000 # controls amount of revision allowed (smaller alpha = more allowed revision)

revision_results = model.barrierGradientRevise(init_seq, log_alpha= log_alpha, outcomeopt_learn_rate = 1.0, max_outcomeopt_iter = 10000, use_adaptive = False)
x_star, z_star, inferred_improvement, outcome_star, reconstruct_init, z_init, outcome_init, avg_sigma_init, edit_dist = revision_results
revision_prob = simulationSeqProbability(x_star)
init_prob = simulationSeqProbability(init_seq[0])
print("Initial sequence:", init_seq)
print("Revised sequence:", x_star)
print("Edit distance:" + str(edit_dist))
print("Initial seq correctly reconstructed? ", reconstruct_init == init_seq[0])
print("Change in latent space:",np.linalg.norm(z_init - z_star))
print("Revised-seq likelihood:", revision_prob, "  Inital-seq likelihood:", init_prob)
print("Actual outcome change:" + str(getSimulationOutcome(x_star) - init_outcome[0]))


# Revise many (num_eval) sequences:
num_eval = 10
eval_seqs, pre_outcomes = generateSimulationData(num_eval)
log_alpha = -10000 # controls amount of revision allowed (smaller alpha = more allowed revision)

post_outcomes = [0 for i in range(num_eval)]
revision_probs = [0 for i in range(num_eval)]
init_probs = [0 for i in range(num_eval)]
num_changed = 0
edit_dists = []
latent_dists = []
init_seqs = [] # only contain changed seqs.
revised_seqs = []
print("revising sequence:", end = " ")
for i in range(num_eval):
    print('revising sequence ', i+1)
    init_seq = [eval_seqs[i]]
    # revision_results = model.fixedGradientRevise(init_seq, outcomeopt_learn_rate = 2.3, max_outcomeopt_iter = 1000)
    revision_results = model.barrierGradientRevise(init_seq, log_alpha= log_alpha, outcomeopt_learn_rate = 1.0, max_outcomeopt_iter = 10000, use_adaptive = False)
    x_star, z_star, inferred_improvement, outcome_star, reconstruct_init, z_init, outcome_init, avg_sigma_init, edit_dist = revision_results
    # x_star, inferred_improvement, outcome_star, outcome_init, edit_dist = model.randomRevise(init_seq, num_candidates = 100, num_edits = 5, print_iter = 0.0, prob_cutoff = 0.1)
    x_star = restrictLength(x_star)
    post_outcomes[i] = getSimulationOutcome(x_star)
    revision_probs[i] = simulationSeqProbability(x_star)
    init_probs[i] = simulationSeqProbability(eval_seqs[i])
    edit_dists.append(edit_dist)
    latent_dists.append(np.linalg.norm(z_init - z_star))
    if x_star != eval_seqs[i]:
        num_changed += 1
        init_seqs.append(eval_seqs[i])
        revised_seqs.append(x_star)

print("Avg outcome-change:", (sum(post_outcomes)-sum(pre_outcomes))/(float(num_eval)*np.std(train_data[1])), " +- ", np.std((np.array(post_outcomes)-np.array(pre_outcomes))/np.std(train_data[1])))
print("Avg revision probability:", sum(revision_probs)/float(num_eval), "  +- ", np.std(revision_probs), "  Avg inital prob:", sum(init_probs)/float(num_eval))
print("Num changed seqs:", num_changed, "  Avg edit distance:", np.mean(edit_dists), "  +- ", np.std(edit_dists))

