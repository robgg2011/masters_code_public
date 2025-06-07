import cartpole_config as config

import jax
import jax.numpy as jnp
import jax.random as random

from functools import partial


def ff_init_agent(random_key):
    """ Features eligibility traces and no adam optimization
    """
    hidden = config.neurons.copy()
    hidden.insert(0, config.input_size + config.num_classes)

    key, *subkeys= random.split(random_key, len(hidden)*3)
        
    W = [random.uniform(subkeys[i-1],(hidden[i], hidden[i-1]), minval=-1) for i in range(1, len(hidden))]  #matrix of weights TO each layer, from layer below. self.W[l][i][j] -> the weight from neuron j in layer l-1 to neuron i in layer l
    
    A = [random.uniform(subkeys[i-1 + len(hidden)], (config.num_classes, hidden[i]), minval=-1) for i in range(1, len(hidden))]

    eligibility = [jnp.zeros_like(W[l]) for l in range(len(W))]

    return [W, A], eligibility


@jax.jit
def generate_negative_data(X, Y, random_key):
    X_neg = X

    Y_neg = jax.random.uniform(random_key, Y.shape, minval=0., maxval=1.) * (1. - Y)
    Y_neg = jax.nn.one_hot(jnp.argmax(Y_neg, axis=1), num_classes=config.num_classes, dtype=jnp.float32)

    return Y_neg, X_neg



@jax.jit
def _ff_loss(W, x, y_type, goodness_threshold):
    z = jnp.maximum(jnp.matmul(x, jnp.transpose(W)), 0)

    delta = jnp.sum(jnp.square(z), axis=1) - goodness_threshold * x.shape[1]
    out = jnp.maximum(delta, 0) - delta * (y_type) + jnp.log(1. + jnp.exp(-jnp.abs(delta)))

    return jnp.mean(out)


@jax.jit
def _symba_loss(W, x, alpha):
    z = jnp.maximum(jnp.matmul(x, jnp.transpose(W)), 0)

    z_pos = z[:config.batch_size]
    z_neg = z[config.batch_size:]
    #z_neg = jnp.flip(z[config.batch_size:], axis=0)

    delta_symba = jnp.sum(jnp.square(z_pos), axis=1) - jnp.sum(jnp.square(z_neg), axis=1)

    out = -jnp.minimum(delta_symba, 0) + jnp.log(1. + jnp.exp(- alpha * jnp.abs(delta_symba)))

    return jnp.mean(out)



@partial(jax.jit, static_argnames=["plasticity", "y_type"])
def agent_ff_process(Xb, Yb, weights, eligibility, step, y_type, random_key, plasticity=True, td_delta=None, *, discount=config.discount, actor_decay=config.actor_decay, actor_lr= config.actor_lr, goodness_threshold=config.goodness_threshold, alpha=config.alpha):
    key, *subkeys = jax.random.split(random_key, 3)
    del random_key

    ### Prepare data ###
    if plasticity:
        Y_neg, X_neg = generate_negative_data(Xb, Yb, subkeys[0])
        Xb = jnp.concat((Xb, X_neg))
        Yb = jnp.concat((Yb, Y_neg))

    W, A = weights

    batch_size = Xb.shape[0]
    num_layers = len(W)
    in_dim = Xb.shape[1]
    neurons = [w.shape[0] for w in W]

    activity = [jnp.zeros((batch_size, n)) for n in neurons]

    if plasticity:
        if y_type == 1:
            Yb_type = jnp.concat((jnp.ones((batch_size//2)), jnp.zeros((batch_size//2))))
        else:
            Yb_type = jnp.concat((jnp.zeros((batch_size//2)), jnp.ones((batch_size//2))))
    else:
        Yb_type = jnp.ones((batch_size))

    total_goodness = jnp.zeros((batch_size))
    
    #input_activity = Xb.at[:, 0:config.num_classes].set(Yb)
    input_activity = jnp.concatenate((Xb, Yb), axis = -1)
    
    for l in range(num_layers):

        if l == 0:
            input = input_activity
        else:
            input = activity[l-1] / jnp.maximum(jnp.linalg.norm(activity[l-1], axis=1, keepdims=True), 1e-5)

        activity[l] = jnp.maximum(jnp.matmul(input, jnp.transpose(W[l])), 0)

        if plasticity:
            match config.training_type:
                case "standard": delta = jax.value_and_grad(_ff_loss)(W[l], input, Yb_type, goodness_threshold)
                case "symba": delta = jax.value_and_grad(_symba_loss)(W[l], input, alpha)

            eligibility[l] = discount * actor_decay * eligibility[l] - (discount ** step) * delta[1] / (delta[0] + 1e-8)
            W[l] = W[l] + actor_lr * eligibility[l] * jnp.abs(td_delta)

        if l != 0:
            total_goodness += jnp.sum(jnp.square(activity[l]), axis=1)


    out = classifier_output(A, activity)

    if plasticity:
        delta = jax.grad(classifier_loss, argnums=1)(Yb, A, activity, Yb_type)

        for l in range(num_layers):
            A[l] = A[l] - config.classifier_learning_rate * delta[l]

    return [W, A], eligibility, out, total_goodness

    

@jax.jit
def classifier_output(weights, activity):
    sum = jnp.zeros((activity[0].shape[0], weights[0].shape[0])) #(batch_size, output_size)
    for l in range(1, len(weights)):
        sum += jnp.matmul(activity[l], jnp.transpose(weights[l]))

    #sum = jnp.matmul(activity[1], jnp.transpose(weights[1]))

    return jax.nn.softmax(sum)


@jax.jit
def classifier_loss(Yb, weights, activity, Yb_type):
    out = classifier_output(weights, activity)

    mod = jnp.expand_dims(Yb_type, axis=1)

    delta = (Yb - out) * mod

    error = jnp.log(1. + jnp.exp(-delta))

    return jnp.mean(error)


"""
@partial(jax.jit, static_argnames=["plasticity"])
def agent_ff_process(X, Y, weights, eligibility, step, y_type, random_key, td_delta=None, plasticity=True):
    key, *subkeys = jax.random.split(random_key, 3)
    del random_key

    ### Prepare data ###
    if plasticity:
        Yb_neg, Xb_neg = generate_negative_data(Xb, Yb, subkeys[0])
        Xb = jnp.concat((Xb, Xb_neg))
        Yb = jnp.concat((Yb, Yb_neg))

    W, A = weights

    batch_size = Xb.shape[0]
    num_layers = len(W)
    in_dim = Xb.shape[1]
    neurons = [w.shape[0] for w in W]

    activity = [jnp.zeros((batch_size, n)) for n in neurons]

    if plasticity:
        Yb_type = jnp.array((y_type, 1-y_type))
    else:
        Yb_type = jnp.array((y_type))


    total_goodness = jnp.zeros((batch_size))
    
    input_activity = jnp.concat((Yb, Xb), axis=-1)

    #print(input_activity.shape)
    
    for l in range(num_layers):

        if l == 0:
            input = input_activity
        else:
            input = activity[l-1] / jnp.maximum(jnp.linalg.norm(activity[l-1], axis=1, keepdims=True), 1e-5)


        activity[l] = jnp.maximum(jnp.matmul(input, jnp.transpose(W[l])), 0)

        if plasticity:
            match config.training_type:
                case "standard": delta = jax.value_and_grad(_ff_loss)(W[l], input, Yb_type)
                case "symba": delta = jax.value_and_grad(_symba_loss)(W[l], input)

            eligibility[l] = config.discount * config.decay * eligibility[l] - (config.discount ** step) * delta[1] / (delta[0] + 1e-8)
            W[l] = W[l] + config.global_learning_rate * eligibility[l] * jnp.abs(td_delta)

        if l != 0:
            total_goodness += jnp.sum(jnp.square(activity[l]), axis=1)


    out = classifier_output(A, activity)

    if plasticity:
        delta = jax.grad(classifier_loss, argnums=1)(Yb, A, activity, Yb_type)

        for l in range(num_layers):
            A[l] = A[l] - config.classifier_learning_rate * delta[l]

    return [W, A], eligibility, out, total_goodness"""