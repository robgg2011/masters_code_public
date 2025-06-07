import csdp_config as config

import jax
import jax.numpy as jnp

from functools import partial


@jax.jit
def generate_negative_data(Xb, Yb, random_key):
    if config.neg_data_type == "random_targets":
        Xb_neg = Xb

        Yb_neg = jax.random.uniform(random_key, Yb.shape, minval=0., maxval=1.) * (1. - Yb)
        Yb_neg = jax.nn.one_hot(jnp.argmax(Yb_neg, axis=1), num_classes=Yb.shape[1], dtype=jnp.float32)

    elif config.neg_data_type == "shuffle_inputs":
        Yb_neg = Yb
        Xb_neg = jax.random.permutation(random_key, Xb)
    
    elif config.neg_data_type == "unique_shuffle_inputs":
        Yb_neg = Yb

        key, *subkeys = jax.random.split(random_key, 4)
        n_index_main = jax.random.permutation(subkeys[0], Xb.shape[0])
        n_index_sec = jax.random.permutation(subkeys[1], Xb.shape[0])

        _y = jnp.argmax(Yb_neg, axis=1, keepdims=True)

        _x = jnp.where(_y != _y[n_index_sec], Xb[n_index_sec], jax.random.permutation(subkeys[2], Xb))
        
        Xb_neg = jnp.where(_y != _y[n_index_main], Xb[n_index_main], _x)
        #Xb_neg = jnp.where(_y != _y[n_index_main], Xb[n_index_main], jax.random.permutation(subkeys[2], Xb))

    return Xb, Yb, Yb_neg, Xb_neg



@jax.jit
def adam_update(grad, state):
    b1=0.9
    b2=0.999
    eps=1e-8

    m, v, step = state
    step = step + 1
    m = (1 - b1) * grad + b1 * m  # First  moment estimate.
    v = (1 - b2) * jnp.square(grad) + b2 * v  # Second moment estimate.
    mhat = m / (1 - b1 ** (step))  # Bias correction.
    vhat = v / (1 - b2 ** (step))

    return config.global_learning_rate * mhat / (jnp.sqrt(vhat) + eps), (m, v, step)



def save_to_file(weights, base_thr, seed=None):
    W, V, M, A, B = weights
    #w_opt, v_opt, m_opt, a_opt, b_opt = optim_state
    thrs, output_threshold = base_thr

    if seed == None: seed = config.SEED

    with open(config.MODEL_PATH + f"{seed}_{config.training_type}_csdp_n{config.neurons[0]}-{config.neurons[1]}_b{config.batch_size}.npy", "wb+") as file:
        for weight in W:
            jnp.save(file, weight)
        for weight in V:
            jnp.save(file, weight)
        for weight in M:
            jnp.save(file, weight)
        for weight in A:
            jnp.save(file, weight)
        for weight in B:
            jnp.save(file, weight)
        for thr in thrs:
            jnp.save(file, thr)
        jnp.save(file, output_threshold)


def load_from_file(seed=None):
    #num_layers = len(config.neurons)
    num_layers = 2

    W = []
    V = []
    M = []
    A = []
    B = []
    thrs = []
    output_threshold = None

    if seed == None: seed = config.SEED

    with open(config.MODEL_PATH + f"{seed}_{config.training_type}_csdp_n{config.neurons[0]}-{config.neurons[1]}_b{config.batch_size}.npy", "rb") as file:
        for _ in range(num_layers):
            W.append(jnp.load(file))
        
        for _ in range(num_layers - 1):
            V.append(jnp.load(file))
        
        for _ in range(num_layers):
            M.append(jnp.load(file))

        for _ in range(num_layers):
            A.append(jnp.load(file))

        for _ in range(num_layers):
            B.append(jnp.load(file))

        for _ in range(num_layers):
            thrs.append(jnp.load(file))
        
        output_threshold = jnp.load(file)
    
    
    w_state = [(jnp.zeros_like(l), jnp.zeros_like(l)) for l in W]
    v_state = [(jnp.zeros_like(l), jnp.zeros_like(l)) for l in V]
    m_state = [(jnp.zeros_like(l), jnp.zeros_like(l)) for l in M]
    a_state = [(jnp.zeros_like(l), jnp.zeros_like(l)) for l in A]
    b_state = [(jnp.zeros_like(l), jnp.zeros_like(l)) for l in B]
    
    return [W, V, M, A, B], [w_state, v_state, m_state, a_state, b_state], [thrs, output_threshold]

