"""
|---------------------|
| Model Configuration |
|---------------------|
"""

#### Global state variables ####
goodness_threshold = 1  #2 
alpha = 0.5 #4  # used by symba algorithm instead of goodness_threshold

global_learning_rate =0.005  # 0.01 


neurons = [2000, 500] 
training_type = "symba" # "standard" or "symba"
batch_size = 500
save_to_file = False
neg_data_type = "random_targets"
# "random_targets"
# "shuffle_inputs" -> 10% chance of same input and target
# "unique_shuffle_inputs" 0.1% chance of same input and target

random_pairing = False

create_umap = False


#### Training configuration ####
SEED = 651 
epochs = 25
verbosity = 0



#### Global data variables ####

data_path = "./data/"
num_classes = 10
input_size = 28*28

VALIDATION_SEED = 2411 #Dont change, will keep validation set constant

MODEL_PATH = f"./models/MNIST/"

SAVE_PATH = f"./ff_standard/trials_data/fashionMNIST/"


# Pytorch data transforms
import torchvision.transforms
from torch import zeros, tensor, flatten

ONE_HOT = torchvision.transforms.Lambda(lambda y: zeros(num_classes).scatter_(0, tensor(y), value=1))
FLATTEN_IMAGE = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), flatten])



### Experimental configuration ###
num_trials = 10
EXPERIMENTAL_SEEDS = [1234, 9876, 1814, 67534, 2824, 42277, 70129, 78, 90397, 74661]