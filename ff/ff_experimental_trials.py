import ff_config as config

import jax.random as random
import jax.numpy as jnp
import jax.nn

import torch
import torchvision.datasets

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from numpy import arange

from tqdm import tqdm

from ff_model import init_model, ff_process

import matplotlib.pyplot as plt


def train(weights, optim_state, key, seed):

    ### Initialize data ###
    torch.manual_seed(seed)

    train_data = torchvision.datasets.FashionMNIST(config.data_path, train=True, transform= config.FLATTEN_IMAGE, target_transform= config.ONE_HOT, download=False)

    train_idx, val_idx = train_test_split(arange(len(train_data)), test_size=10_000, shuffle=True, random_state=config.VALIDATION_SEED, stratify=train_data.targets)

    train = DataLoader(Subset(train_data, train_idx), batch_size=config.batch_size, shuffle=True)
    validate = DataLoader(Subset(train_data, val_idx), batch_size=2*config.batch_size, shuffle=False)

    num_batches = len(train)

    training_accuracy = []
    validation_accuracy = []
    validation_accuracy_classifier = []


    ### Run Experiments ###
    best_params = [weights, 0]

    for epoch in range(config.epochs):

        ### Train ###
        n_batch = 0
        total = 0
        N = 0

        total_nll = 0
        for x, y in train:
            x = jnp.array(x)
            y = jnp.array(y)

            key, subkey = random.split(key)
            nweights, noptim_state, out, _ = ff_process(x, y, weights, optim_state, subkey, plasticity=True)

            weights = nweights
            optim_state = noptim_state
            
            batch_acc = jnp.sum(jnp.argmax(out[:config.batch_size], axis=1) == jnp.argmax(y, axis=1))

            total += batch_acc
            N += config.batch_size

            n_batch += 1

            print(f"Epoch {epoch} ({n_batch}/{num_batches}): Training Accuracy = {(100*total / N):.3f}", end="\r")

        training_accuracy.append(100*total / N)  


        ### Validate classifier ###
        total = 0
        N = 0

        for x, y in validate:
            x = jnp.array(x)
            y = jnp.array(y)

            key, subkey = random.split(key)
            _, _, out, goodness = ff_process(x, jnp.zeros_like(y), weights, optim_state, subkey, plasticity=False)

            total += jnp.sum(jnp.argmax(out, axis=1) == jnp.argmax(y, axis=1))

            N += 2*config.batch_size

        validation_accuracy_classifier.append(100*total / N)

        _class_acc = 100*total / N


        ### Validate goodness ###
        total = 0
        N = 0
        for x, y in validate:
            x = jnp.array(x)
            y = jnp.array(y)

            best_classes = (-jnp.ones(x.shape[0]), jnp.full((x.shape[0]), -jnp.inf))
            for i in range(config.num_classes):
                key, subkey = random.split(key)
                _, _, _, goodness = ff_process(x, jax.nn.one_hot(jnp.full((x.shape[0]), i), config.num_classes), weights, optim_state, subkey, plasticity=False)


                best_classes = (jnp.where(goodness > best_classes[1], i, best_classes[0]), jnp.maximum(goodness, best_classes[1]))

            total += jnp.sum(best_classes[0] == jnp.argmax(y, axis=1))

            N += 2*config.batch_size
            
        validation_accuracy.append(100*total / N)
        
        if total / N > best_params[1]:
            best_params = [weights, total / N]

        print(f"Epoch {epoch} Validation Accuracy - Goodness: {(100*total / N):.3f} / Classifier: {_class_acc:.3f}")
    
    return best_params[0], training_accuracy, validation_accuracy, validation_accuracy_classifier



def test(weights, optim_state, key):
    test_data = torchvision.datasets.FashionMNIST(config.data_path, train=False, transform= config.FLATTEN_IMAGE, target_transform= config.ONE_HOT, download=False)
    
    test = torch.utils.data.DataLoader(test_data, batch_size=2*config.batch_size, shuffle=True)

    ### Test classifier ###
    total = 0
    N = 0
    
    for x, y in test:
        x = jnp.array(x)
        y = jnp.array(y)

        key, subkey = random.split(key)
        _, _, out, goodness = ff_process(x, jnp.zeros_like(y), weights, optim_state, subkey, plasticity=False)

        total += jnp.sum(jnp.argmax(out, axis=1) == jnp.argmax(y, axis=1))

        N += 2*config.batch_size

    class_out = total / N

    print(f"Testing Accuracy (Classifier): {(100*class_out):.3f}")

    ### Test goodness ###
    total = 0
    N = 0

    total_nll = 0
    for x, y in test:
        x = jnp.array(x)
        y = jnp.array(y)

        best_classes = (-jnp.ones(x.shape[0]), jnp.full((x.shape[0]), -jnp.inf))
        for i in range(config.num_classes):
            key, subkey = random.split(key)
            _, _, _, goodness = ff_process(x, jax.nn.one_hot(jnp.full((x.shape[0]), i), config.num_classes), weights, optim_state, subkey, plasticity=False)


            best_classes = (jnp.where(goodness > best_classes[1], i, best_classes[0]), jnp.maximum(goodness, best_classes[1]))

        total += jnp.sum(best_classes[0] == jnp.argmax(y, axis=1))

        N += 2*config.batch_size

    print(f"Testing Accuracy (Goodness): {(100*total / N):.3f}")

    return total / N, class_out



def save_to_file(test_acc, class_test_acc, training_acc, validation_acc, classifier_acc):
    with open(config.SAVE_PATH + f"trials_{config.training_type}_FF_{config.neg_data_type}_n{config.neurons[0]}-{config.neurons[1]}_b{config.batch_size}.npy", "wb+") as file:
        jnp.save(file, test_acc)
        jnp.save(file, class_test_acc)
        jnp.save(file, training_acc)
        jnp.save(file, validation_acc)
        jnp.save(file, classifier_acc)


def read_file():
    with open(config.SAVE_PATH + f"trials_{config.training_type}_FF_{config.neg_data_type}_n{config.neurons[0]}-{config.neurons[1]}_b{config.batch_size}.npy", "rb") as file:
        _test = jnp.load(file)
        _test_class = jnp.load(file)
        _train = jnp.load(file)
        _val = jnp.load(file)
        _class = jnp.load(file)
    
    return _test, _test_class, _train, _val, _class


def main():

    record = []
    record_class = []
    train_convergence = []
    validation_convergence = []
    classifier_convergence = []
    
    for i in tqdm(range(config.num_trials)):
        #print(f"------------- Trial {i} -------------")

        key, *subkeys = random.split(random.key(config.EXPERIMENTAL_SEEDS[i]), 4)

        weights, optim_state = init_model(subkeys[0])
        

        train_weights, training_accuracy, validation_accuracy, validation_accuracy_classifier = train(weights, optim_state, subkeys[1], config.EXPERIMENTAL_SEEDS[i])
        
        train_convergence.append(training_accuracy)
        validation_convergence.append(validation_accuracy)
        classifier_convergence.append(validation_accuracy_classifier)

        acc, class_acc = test(train_weights, optim_state, subkeys[2])

        record.append(acc)
        record_class.append(class_acc)

    record = jnp.asarray(record)
    print(record)

    print(f"Large Fashion FF {config.training_type} ({config.neg_data_type}): Mean acc = {(100*jnp.mean(record)):.3f} Std = {jnp.std(100*record):.3f}")

    print(jnp.asarray(record_class))
    print(f"Classifier: Mean acc = {(100*jnp.mean(jnp.asarray(record_class))):.3f} Std = {jnp.std(100*jnp.asarray(record_class)):.3f}")


    train_plot = jnp.mean(jnp.asarray(train_convergence), axis=0)
    val_plot = jnp.mean(jnp.asarray(validation_convergence), axis=0)
    class_plot = jnp.mean(jnp.asarray(classifier_convergence), axis=0)


    ### Display training metrics ###
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))

    axs[0].plot(train_plot)
    axs[0].set_title("Average Training Accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Accuracy")

    axs[1].plot(val_plot, label="Goodness")
    axs[1].plot(class_plot, label="Classifier")
    axs[1].set_title("Validation Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")

    plt.tight_layout()
    plt.legend()
    plt.savefig(f"fashion_large_{config.training_type}_ff_{config.neg_data_type}_convergence_plot.png", bbox_inches="tight")
    plt.show()


    save_to_file(jnp.asarray(record), jnp.asarray(record_class), jnp.asarray(train_convergence), jnp.asarray(validation_convergence), jnp.asarray(classifier_convergence))



if __name__ == "__main__":
    main()
