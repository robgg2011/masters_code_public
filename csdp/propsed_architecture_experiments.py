import csdp_config as config

import jax.random as random
import jax.numpy as jnp
import jax.nn

import torch
import torchvision.datasets

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from numpy import arange

from tqdm import tqdm, trange

from csdp_model import init_model, csdp_process
from csdp_functional_library import save_to_file

import matplotlib.pyplot as plt


def train(weights, optim_state, base_thresholds, key, seed):

    ### Initialize data ###
    torch.manual_seed(seed)

    train_data = torchvision.datasets.MNIST(config.data_path, train=True, transform= config.FLATTEN_IMAGE, target_transform= config.ONE_HOT, download=False)

    train_idx, val_idx = train_test_split(arange(len(train_data)), test_size=10_000, shuffle=True, random_state=config.VALIDATION_SEED, stratify=train_data.targets)

    train = DataLoader(Subset(train_data, train_idx), batch_size=config.batch_size, shuffle=True)
    validate = DataLoader(Subset(train_data, val_idx), batch_size=2*config.batch_size, shuffle=False)

    num_batches = len(train)

    training_accuracy = []
    validation_accuracy = []

    ### Run Experiments ###
    best_params = [weights, 0]

    for epoch in range(config.epochs):

        ### Train ###
        n_batch = 0
        total = 0
        N = 0

        total_nll = 0
        for x, y in train:
            thr = [[base_thresholds[0][0] +0, base_thresholds[0][1] +0], base_thresholds[1] +0]

            x = jnp.array(x)
            y = jnp.array(y)

            key, subkey = random.split(key)
            nweights, noptim_state, out, _, _ = csdp_process(x, y, weights, optim_state, thr, subkey, plasticity=True)

            weights = nweights
            optim_state = noptim_state
            
            total += jnp.sum(jnp.argmax(out[:config.batch_size], axis=1) == jnp.argmax(y, axis=1))
            N += config.batch_size

            n_batch += 1

            print(f"Epoch {epoch} ({n_batch}/{num_batches}): Accuracy = {(100*total / N):.3f}", end="\r")

        training_accuracy.append(100*total / N) 


        ### Validate ###
        total = 0
        N = 0

        total_nll = 0
        for x, y in validate:

            thr = [[base_thresholds[0][0] +0, base_thresholds[0][1] +0], base_thresholds[1] +0]

            x = jnp.array(x)
            y = jnp.array(y)

            key, subkey = random.split(key)
            _, _, out, count, goodness = csdp_process(x, jnp.zeros_like(y), weights, optim_state, thr, subkey, plasticity=False)

            total += jnp.sum(jnp.argmax(out, axis=1) == jnp.argmax(y, axis=1))

            N += 2*config.batch_size


            p = out
            offset=1e-7
            p_ = jnp.clip(p, offset, 1.0 - offset)
            loss = -(y * jnp.log(p_))
            nll = jnp.sum(loss) #/(y_true.shape[0] * 1.0)
            total_nll += nll

        validation_accuracy.append(100*total / N)

        if total / N > best_params[1]:
            best_params = [weights, total / N]

        print(f"Epoch {epoch} Validation Accuracy: {(100*total / N):.3f} NLL: {(total_nll/(20*config.batch_size)):.5f}")
    
    if config.save_to_file:
        save_to_file(best_params[0], base_thresholds, seed)
    
    return best_params[0], training_accuracy, validation_accuracy



def test(weights, optim_state, base_thresholds, key):
    test_data = torchvision.datasets.MNIST(config.data_path, train=False, transform= config.FLATTEN_IMAGE, target_transform= config.ONE_HOT, download=False)
    
    test = torch.utils.data.DataLoader(test_data, batch_size=2*config.batch_size, shuffle=True)

    ### Test ###
    total = 0
    N = 0
    for x, y in test:

        thr = [[base_thresholds[0][0] +0, base_thresholds[0][1] +0], base_thresholds[1] +0]

        x = jnp.array(x)
        y = jnp.array(y)

        key, subkey = random.split(key)
        weights, _, out, count, goodness = csdp_process(x, jnp.zeros_like(y), weights, optim_state, thr, subkey, plasticity=False)

        total += jnp.sum(jnp.argmax(out, axis=1) == jnp.argmax(y, axis=1))

        N += 2*config.batch_size

    _acc = total / N

    ### Test goodness ###
    total = 0
    N = 0

    total_nll = 0
    for x, y in test:
        thr = [[base_thresholds[0][0] +0, base_thresholds[0][1] +0], base_thresholds[1] +0]

        x = jnp.array(x)
        y = jnp.array(y)

        best_classes = (-jnp.ones(x.shape[0]), jnp.full((x.shape[0]), -jnp.inf))
        for i in range(config.num_classes):
            key, subkey = random.split(key)
            _, _, _, _, goodness = csdp_process(x, jax.nn.one_hot(jnp.full((x.shape[0]), i), config.num_classes), weights, optim_state, thr, subkey, plasticity=False)

            best_classes = (jnp.where(goodness > best_classes[1], i, best_classes[0]), jnp.maximum(goodness, best_classes[1]))

        total += jnp.sum(best_classes[0] == jnp.argmax(y, axis=1))

        N += 2*config.batch_size

    print(f"Testing Accuracy: {100*_acc:.3f} / (Goodness) {(100*total / N):.3f}")

    return _acc, total / N




def main():

    record = []
    gn_record = []

    train_convergence = []
    validation_convergence = []
    
    for i in trange(config.num_trials):
        #print(f"------------- Trial {i} -------------")

        key, *subkeys = random.split(random.key(config.EXPERIMENTAL_SEEDS[i]), 4)

        weights, optim_state, base_thresholds = init_model(subkeys[0])
        

        train_weights, training_accuracy, validation_accuracy = train(weights, optim_state, base_thresholds, subkeys[1], config.EXPERIMENTAL_SEEDS[i])
        
        train_convergence.append(training_accuracy)
        validation_convergence.append(validation_accuracy)


        acc, gn_acc = test(train_weights, optim_state, base_thresholds, subkeys[2])

        record.append(acc)
        gn_record.append(gn_acc)

    record = jnp.asarray(record)
    print(record)
    print(f"50 Epochs CSDP {config.training_type} ({config.neg_data_type}): Mean acc = {(100*jnp.mean(record)):.3f} Std = {jnp.std(100*record):.3f}")

    print(jnp.asarray(gn_record))
    print(f"(Goodness) Mean acc = {(100*jnp.mean(jnp.asarray(gn_record))):.3f} Std = {jnp.std(100*jnp.asarray(gn_record)):.3f}")



    train_plot = jnp.mean(jnp.asarray(train_convergence), axis=0)
    val_plot = jnp.mean(jnp.asarray(validation_convergence), axis=0)


    ### Display training metrics ###
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))

    axs[0].plot(train_plot)
    axs[0].set_title("Average Training Accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Accuracy")

    axs[1].plot(val_plot)
    axs[1].set_title("Average Validation Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig(f"50_epochs_{config.training_type}_csdp_{config.neg_data_type}_convergence_plot.png", bbox_inches='tight')




if __name__ == "__main__":
    main()
