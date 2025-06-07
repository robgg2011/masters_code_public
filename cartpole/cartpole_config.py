"""
|------------------|
| RL Configuration |
|------------------|
"""


#### Global env variables ####

# CSDP params #
training_stimulus_time = 90
testing_stimulus_time = 90
integration_constant = 3

SPIKE_THRESHOLD = 0.055
SPIKE_THRESHOLD_JITTER = 0.025
excitatory_resistance = 0.1
inhibitory_resistance = 0.01
tau_m = 100
lambda_v = 0.001
tau_tr = 13

lambda_d = 0.00001 #0.00001 #0.00001 


# General params #
goodness_threshold = 1 #2
alpha = 1.2 # used by symba algorithm instead of goodness_threshold

classifier_learning_rate = 0.001

training_type = "standard" # "standard" or "symba"
batch_size = 1
save_to_file = False


actor_lr= 0.00005 #0.001
critic_lr= 0.001 #0.001

hidden = 50
neurons = [500, 200]

actor_type = "ff"
# "standard"
# "ff"
# "csdp"


discount=0.995
C=2
actor_decay = 0.1 #0.9
critic_decay = 0.9


num_classes = 2
input_size = 4

n_episodes = 1000


SEED = 3456234 #651


#### Hyper parameter search and trial variables ####
n_train_episodes = 250
n_test_episodes = 50


#### Experimental trials variables ####
num_trials = 20
EXPERIMENTAL_SEEDS = [1234, 9876, 1814, 67534, 2824, 42277, 70129, 78, 90397, 74661, 54, 366, 2547, 8465, 282, 824524, 245613, 23462, 54633, 9835]
