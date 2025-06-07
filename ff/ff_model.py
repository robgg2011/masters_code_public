import ff_config as config

import jax
import jax.numpy as jnp
import jax.random as random

from functools import partial


def init_model(random_key):
    hidden = config.neurons.copy()
    hidden.insert(0, config.input_size)

    key, *subkeys= random.split(random_key, len(hidden)*3)
        
    W = [random.uniform(subkeys[i-1],(hidden[i], hidden[i-1]), minval=-1) for i in range(1, len(hidden))]  #matrix of weights TO each layer, from layer below. self.W[l][i][j] -> the weight from neuron j in layer l-1 to neuron i in layer l
    
    A = [random.uniform(subkeys[i-1 + len(hidden)], (config.num_classes, hidden[i]), minval=-1) for i in range(1, len(hidden))]

    w_state = [(jnp.zeros_like(l), jnp.zeros_like(l), 0) for l in W]
    a_state = [(jnp.zeros_like(l), jnp.zeros_like(l), 0) for l in A]

    return [W, A], [w_state, a_state]


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

    return Yb_neg, Xb_neg



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


@jax.jit
def _ff_loss(W, x, y_type):
    z = jnp.maximum(jnp.matmul(x, jnp.transpose(W)), 0)

    delta = jnp.sum(jnp.square(z), axis=1) - config.goodness_threshold * z.shape[1]
    #delta = jnp.mean(jnp.square(z), axis=1) - config.goodness_threshold # set goodness threshold -> 2

    #delta = jnp.sum(jnp.square(z), axis=1) - config.goodness_threshold
    out = jnp.maximum(delta, 0) - delta * (y_type) + jnp.log(1. + jnp.exp(-jnp.abs(delta)))

    #delta = y_type * (jnp.mean(jnp.square(z), axis=1) - config.goodness_threshold) + (1-y_type) * (config.goodness_threshold - jnp.mean(jnp.square(z), axis=1))
    #out = jnp.log(1. + jnp.exp(-delta))

    return jnp.mean(out)



@jax.jit
def _symba_loss(W, x):
    z = jnp.maximum(jnp.matmul(x, jnp.transpose(W)), 0)

    z_pos = z[:config.batch_size]
    z_neg = z[config.batch_size:]
    
    if config.random_pairing:
        z_neg = jnp.flip(z[config.batch_size:], axis=0) # NOTE effectively "randomizes" sample pairing

    delta_symba = jnp.sum(jnp.square(z_pos), axis=1) - jnp.sum(jnp.square(z_neg), axis=1)

    #out = jnp.log(1. + jnp.exp(- config.alpha * delta_symba)) # This is unstable
    
    out = -jnp.minimum(delta_symba, 0) + jnp.log(1. + jnp.exp(- config.alpha * jnp.abs(delta_symba)))

    return jnp.mean(out)



@partial(jax.jit, static_argnames=["plasticity", "record_latent"])
def ff_process(Xb, Yb, weights, optim_state, random_key, plasticity=True, record_latent=False):
    key, *subkeys = jax.random.split(random_key, 3)
    del random_key

    ### Prepare data ###
    if plasticity:
        Yb_neg, Xb_neg = generate_negative_data(Xb, Yb, subkeys[0])
        Xb = jnp.concat((Xb, Xb_neg))
        Yb = jnp.concat((Yb, Yb_neg))

    W, A = weights
    w_opt, a_opt = optim_state

    batch_size = Xb.shape[0]
    num_layers = len(W)
    in_dim = Xb.shape[1]
    neurons = [w.shape[0] for w in W]

    activity = [jnp.zeros((batch_size, n)) for n in neurons]

    if plasticity:
        Yb_type = jnp.concat((jnp.ones((batch_size//2)), jnp.zeros((batch_size//2))))
    else:
        Yb_type = jnp.ones((batch_size))

    total_goodness = jnp.zeros((batch_size))
    
    input_activity = Xb.at[:, 0:10].set(Yb)
    
    for l in range(num_layers):

        if l == 0:
            input = input_activity
        else:
            input = activity[l-1] / jnp.maximum(jnp.linalg.norm(activity[l-1], axis=1, keepdims=True), 1e-5)

        #activation = jnp.matmul(input, jnp.transpose(W[l]))
        #activation = jnp.matmul(input, jnp.transpose(W[l])) #+ jnp.matmul(Yb, jnp.transpose(B[l]))
        activity[l] = jnp.maximum(jnp.matmul(input, jnp.transpose(W[l])), 0)

        #activity[l] = activation / jnp.maximum(jnp.linalg.norm(activation, axis=1, keepdims=True), 1e-3)
        if record_latent:
            if l == 0:
                #latent_distribtuion = activity[l]
                pass
            elif l == 1:
                latent_distribtuion = activity[l]
            else:
                latent_distribtuion = jnp.concatenate((latent_distribtuion, activity[l]), axis=-1)

        if plasticity:
            match config.training_type:
                case "standard": delta = jax.grad(_ff_loss)(W[l], input, Yb_type)
                case "symba": delta = jax.grad(_symba_loss)(W[l], input)

            dW, _temp = adam_update(delta, w_opt[l])
            #W[l] = jnp.clip(W[l] - dW, -1, 1)
            W[l] = W[l] - dW
            w_opt[l] = _temp

        if l != 0:
            total_goodness += jnp.sum(jnp.square(activity[l]), axis=1)


    out = classifier_output(A, activity)

    if plasticity:
        delta = jax.grad(classifier_loss, argnums=1)(Yb, A, activity, Yb_type)

        for l in range(num_layers):
            dA, _temp = adam_update(delta[l], a_opt[l])
            #A[l] = jnp.clip(A[l] - dA, -1, 1)
            A[l] = A[l] - dA
            a_opt[l] = _temp

    if record_latent:
        return [W, A], [w_opt, a_opt], out, total_goodness, latent_distribtuion

    return [W, A], [w_opt, a_opt], out, total_goodness

    

@jax.jit
def classifier_output(weights, activity):
    """sum = jnp.zeros((activity[0].shape[0], weights[0].shape[0])) #(batch_size, output_size)
    for l in range(len(weights)):
        sum += jnp.matmul(activity[l], jnp.transpose(weights[l]))"""

    sum = jnp.matmul(activity[1], jnp.transpose(weights[1]))

    return jax.nn.softmax(sum)


@jax.jit
def classifier_loss(Yb, weights, activity, Yb_type):
    out = classifier_output(weights, activity)

    mod = jnp.expand_dims(Yb_type, axis=1)

    delta = (Yb - out) * mod

    error = jnp.log(1. + jnp.exp(-delta))

    return jnp.mean(error)

