"""
|---------------------|
| Model Configuration |
|---------------------|
"""

#### Global state variables ####
training_stimulus_time = 150
testing_stimulus_time = 150
integration_constant = 3

SPIKE_THRESHOLD = 0.055
SPIKE_THRESHOLD_JITTER = 0.025
excitatory_resistance = 0.1
inhibitory_resistance = 0.01 #0.035
tau_m = 100
lambda_v = 0.001
tau_tr = 13
gamma = 0.05 # Used for non-linear trace activation

goodness_threshold = 10
alpha = 0.7 # 1.5 # used by symba algorithm instead of goodness_threshold

lambda_d = 0.00005 
global_learning_rate =0.002 


neurons = [2000, 500]
training_type = "standard" # "standard" or "symba"
batch_size = 500
save_to_file = False
neg_data_type = "random_targets"
# "random_targets"
# "shuffle_inputs" -> 10% chance of same input and target
# "unique_shuffle_inputs" 0.1% chance of same input and target

random_pairing = False

create_umap = False


#### Training configuration ####
SEED = 74661
epochs = 25
verbosity = 0



#### Global data variables ####

data_path = "./data/"
num_classes = 10
input_size = 28*28

VALIDATION_SEED = 2411 #Dont change, will keep validation set constant

MODEL_PATH = f"./models/"


# Pytorch data transforms
import torchvision.transforms
from torch import zeros, tensor, flatten

ONE_HOT = torchvision.transforms.Lambda(lambda y: zeros(num_classes).scatter_(0, tensor(y), value=1))
FLATTEN_IMAGE = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), flatten])