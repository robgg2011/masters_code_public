import csdp_config as config

import torch
import torchvision.datasets

import jax.numpy as jnp
import jax.nn
from jax import random

from csdp_model import init_model, csdp_process
from csdp_functional_library import load_from_file

from tqdm import tqdm


from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from numpy import arange

"""
Model is retrieved from file based on config specification:
    - Learning type, neurons and batch size specify model

"""

def main():

    ### Initialize data ###
    #torch.manual_seed(config.SEED)

    test_data = torchvision.datasets.MNIST(config.data_path, train=False, transform= config.FLATTEN_IMAGE, target_transform= config.ONE_HOT, download=False)
    
    test = torch.utils.data.DataLoader(test_data, batch_size=2*config.batch_size, shuffle=True)
    

    ### Initialize model ###
    key, subkey = random.split(random.key(config.SEED), 2)

    weights, optim_state, base_thresholds = load_from_file()


    ### Test ###
    total = 0
    goodness_total = 0
    N = 0
    for x, y in tqdm(test):
        
        ### Classifier ###
        thr = [[base_thresholds[0][0] +0, base_thresholds[0][1] +0], base_thresholds[1] +0]

        x = jnp.array(x)
        y = jnp.array(y)

        key, subkey = random.split(key)
        weights, optim_state, out, count, goodness = csdp_process(x, jnp.zeros_like(y), weights, optim_state, thr, subkey, plasticity=False)

        total += jnp.sum(jnp.argmax(out, axis=1) == jnp.argmax(y, axis=1))

        N += 2*config.batch_size


        ### Goodness ###
        thr = [[base_thresholds[0][0] +0, base_thresholds[0][1] +0], base_thresholds[1] +0]

        x = jnp.array(x)
        y = jnp.array(y)

        best_classes = (-jnp.ones(x.shape[0]), jnp.full((x.shape[0]), -jnp.inf))
        for i in range(config.num_classes):
            key, subkey = random.split(key)
            _, _, _, _, goodness = csdp_process(x, jax.nn.one_hot(jnp.full((x.shape[0]), i), config.num_classes), weights, optim_state, thr, subkey, plasticity=False)

            best_classes = (jnp.where(goodness > best_classes[1], i, best_classes[0]), jnp.maximum(goodness, best_classes[1]))

        goodness_total += jnp.sum(best_classes[0] == jnp.argmax(y, axis=1))


    print(f"Accuracy: Classifier - {(100*total / N):.3f}  Goodness - {(100*goodness_total / N):.3f}")

    # Implement metrics for checking performance


if __name__ == "__main__":
    main()