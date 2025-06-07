import csdp_config as config

import torch
import torchvision.datasets
from torch.utils.data import DataLoader, Subset

import jax.numpy as jnp
from jax import random
import jax.nn

from numpy import arange

from sklearn.model_selection import train_test_split

from csdp_model import init_model, csdp_process

from csdp_functional_library import save_to_file

from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import umap

#jax.config.update("jax_debug_nans", True)


def main():

    ### Initialize data ###
    torch.manual_seed(config.SEED)

    train_data = torchvision.datasets.MNIST(config.data_path, train=True, transform= config.FLATTEN_IMAGE, target_transform= config.ONE_HOT, download=False)

    train_idx, val_idx = train_test_split(arange(len(train_data)), test_size=10_000, shuffle=True, random_state=config.VALIDATION_SEED, stratify=train_data.targets)

    train = DataLoader(Subset(train_data, train_idx), batch_size=config.batch_size, shuffle=True)
    validate = DataLoader(Subset(train_data, val_idx), batch_size=2*config.batch_size, shuffle=False)

    num_batches = len(train)

    
    ### Initialize model ###
    key, subkey = random.split(random.key(config.SEED), 2)

    weights, optim_state, base_thresholds = init_model(subkey)

    best_params = [weights, 0]


    training_accuracy = []
    validation_accuracy = []
    batch_accuracy = []
    goodness_accuracy = []

    for epoch in tqdm(range(config.epochs)):

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
            
            batch_acc = jnp.sum(jnp.argmax(out[:config.batch_size], axis=1) == jnp.argmax(y, axis=1))

            total += batch_acc
            N += config.batch_size

            n_batch += 1
            batch_accuracy.append(100*batch_acc / config.batch_size)

            print(f"Epoch {epoch} ({n_batch}/{num_batches}): Training Accuracy = {(100*total / N):.3f}", end="\r")
            
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

        validation_accuracy.append(float(100*total / N))
        
        if total / N > best_params[1]:
            best_params = [weights, total / N]

        print(f"Epoch {epoch} Validation Accuracy: {(100*total / N):.3f} NLL: {(total_nll/(20*config.batch_size)):.5f}")
        


        ### Validate goodness ###
        total = 0
        N = 0

        total_nll = 0
        for x, y in validate:
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

        goodness_accuracy.append(100*total / N)
        
        if total / N > best_params[1]:
            best_params = [weights, total / N]

        print(f"Epoch {epoch} Goodness Accuracy: {(100*total / N):.3f} NLL: {(total_nll/(20*config.batch_size)):.5f}")
    
    if config.save_to_file:
       save_to_file(best_params[0], base_thresholds)


    ### Display training metrics ###

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))

    axs[0].plot(training_accuracy)
    axs[0].set_title("Training Accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Accuracy")

    axs[1].plot(validation_accuracy)
    axs[1].set_title("Validation Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")

    axs[2].plot(goodness_accuracy)
    axs[2].set_title("Goodness accuracy")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Accuracy")

    """axs[3].plot(batch_accuracy)
    axs[3].set_title("Batch validation accuracy")
    axs[3].set_xlabel("Batch")
    axs[3].set_ylabel("Accuracy")"""

    plt.tight_layout()
    plt.show()

    

    if config.create_umap:
        ### Test ###
        test_data = torchvision.datasets.MNIST(config.data_path, train=False, transform= config.FLATTEN_IMAGE, target_transform= config.ONE_HOT, download=False)
        
        test = torch.utils.data.DataLoader(test_data, batch_size=2*config.batch_size, shuffle=True)

        total = 0
        N = 0

        latent_distribution = None
        targets = None
        outputs = None

        for x, y in test:
            thr = [[base_thresholds[0][0] +0, base_thresholds[0][1] +0], base_thresholds[1] +0]

            x = jnp.array(x)
            y = jnp.array(y)

            key, subkey = random.split(key)
            _, _, out, count, goodness, latent_act = csdp_process(x, jnp.zeros_like(y), weights, optim_state, thr, subkey, plasticity=False, record_latent=True)

            total += jnp.sum(jnp.argmax(out, axis=1) == jnp.argmax(y, axis=1))

            N += 2*config.batch_size

            if latent_distribution is None:
                latent_distribution = latent_act
            else:
                latent_distribution = jnp.concatenate((latent_distribution, latent_act), axis=0)

            if targets is None:
                targets = jnp.argmax(y, axis=1)
            else:
                targets = jnp.concatenate((targets, jnp.argmax(y, axis=1)), axis=0)

            if outputs is None:
                outputs = jnp.argmax(out, axis=1)
            else:
                outputs = jnp.concatenate((outputs, jnp.argmax(out, axis=1)), axis=0)
                
            N += 2*config.batch_size

        print(f"TEST Accuracy: {(100*total / N):.3f}")

        

        reducer = umap.UMAP(random_state=config.SEED)
        reducer.fit(latent_distribution)

        embedding = reducer.transform(latent_distribution)

        plt.scatter(embedding[:, 0], embedding[:, 1], c=targets, cmap='Spectral', s=5)
        plt.gca().set_aspect('equal', 'datalim')
        plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
        plt.title('CSDP FashionMNIST UMAP projection of latent representation (target labels)', fontsize=24)
        plt.show()


        plt.scatter(embedding[:, 0], embedding[:, 1], c=outputs, cmap='Spectral', s=5)
        plt.gca().set_aspect('equal', 'datalim')
        plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
        plt.title('CSDP UMAP projection of latent representation (output labels)', fontsize=24)
        plt.show()


if __name__ == "__main__":
    main()