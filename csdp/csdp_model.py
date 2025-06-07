import csdp_config as config

import jax
import jax.numpy as jnp
from jax import random

from functools import partial

from csdp_functional_library import generate_negative_data, adam_update


def init_model(random_key):
    hidden = config.neurons.copy()
    hidden.insert(0, config.input_size)

    key, *subkeys= random.split(random_key, len(hidden)*5)
        
    W = [random.uniform(subkeys[i-1],(hidden[i], hidden[i-1]), minval=-1) for i in range(1, len(hidden))]  #matrix of weights TO each layer, from layer below. self.W[l][i][j] -> the weight from neuron j in layer l-1 to neuron i in layer l
    V = [random.uniform(subkeys[i-1 + len(hidden)],(hidden[i], hidden[i+1]), minval=-1) for i in range(1, len(hidden)-1)] 
    M = [random.uniform(subkeys[i-1 + len(hidden)*2 - 1],(hidden[i], hidden[i]), minval= 0) for i in range(1, len(hidden))]

    A = [random.uniform(subkeys[i-1+ len(hidden)*3 - 1],(config.num_classes, hidden[i]), minval=-1) for i in range(1, len(hidden))]
    B = [random.uniform(subkeys[i-1+ len(hidden)*4 - 1],(hidden[i], config.num_classes), minval=-1) for i in range(1, len(hidden))]

    w_state = [(jnp.zeros_like(l), jnp.zeros_like(l), 0) for l in W]
    v_state = [(jnp.zeros_like(l), jnp.zeros_like(l), 0) for l in V]
    m_state = [(jnp.zeros_like(l), jnp.zeros_like(l), 0) for l in M]
    a_state = [(jnp.zeros_like(l), jnp.zeros_like(l), 0) for l in A]
    b_state = [(jnp.zeros_like(l), jnp.zeros_like(l), 0) for l in B]

    num_layers = len(config.neurons)

    maxthr = config.SPIKE_THRESHOLD+config.SPIKE_THRESHOLD_JITTER
    minthr = config.SPIKE_THRESHOLD-config.SPIKE_THRESHOLD_JITTER

    key, *thrkeys = jax.random.split(key, num_layers+1)
    thresholds = [jax.random.uniform(thrkeys[i], (1, config.neurons[i]), minval=minthr, maxval=maxthr) for i in range(num_layers)] 

    key, out_thrkey = jax.random.split(key)
    output_threshold = jax.random.uniform(out_thrkey, (1, config.num_classes), minval=minthr, maxval=maxthr)

    return [W, V, M, A, B], [w_state, v_state, m_state, a_state, b_state], [thresholds, output_threshold]



@jax.jit
def _contrastive_loss(z, y_type):
    delta = jnp.sum(jnp.square(z), axis=1) - config.goodness_threshold

    out = jnp.maximum(delta, 0) - delta * (y_type) + jnp.log(1. + jnp.exp(-jnp.abs(delta)))

    return jnp.mean(out)


@jax.jit
def _symba_loss(z):
    z_pos = z[:config.batch_size]
    z_neg = z[config.batch_size:]
    
    if config.random_pairing:
        z_neg = jnp.flip(z[config.batch_size:], axis=0) # NOTE effectively randomizes pairing

    delta_symba = jnp.sum(jnp.square(z_pos), axis=1) - jnp.sum(jnp.square(z_neg), axis=1)

    #out = jnp.log(1. + jnp.exp(- config.alpha * delta_symba)) # This is unstable
    
    out = -jnp.minimum(delta_symba, 0) + jnp.log(1. + jnp.exp(- config.alpha * jnp.abs(delta_symba)))

    return jnp.mean(out)



@partial(jax.jit, static_argnames=["plasticity", "record_latent"])
def csdp_process(Xb, Yb, weights, optim_state, thresholds, random_key, plasticity=True, record_latent=False):
    key, *subkeys = jax.random.split(random_key, 3)
    del random_key

    dt = config.integration_constant
    if plasticity:
        T = config.training_stimulus_time // dt
    else:
        T = config.testing_stimulus_time // dt
    

    ### Prepare data ###
    if plasticity:
        Xb, Yb, Yb_neg, Xb_neg = generate_negative_data(Xb, Yb, subkeys[0])
        Xb = jnp.concat((Xb, Xb_neg))
        Yb = jnp.concat((Yb, Yb_neg))

    W, V, M, A, B = weights
    w_opt, v_opt, m_opt, a_opt, b_opt = optim_state

    batch_size = Xb.shape[0]
    num_layers = len(W)
    in_dim = Xb.shape[1]
    neurons = [w.shape[0] for w in W]

    voltages = [jnp.zeros((batch_size, n)) for n in neurons]
    spikes = [jnp.zeros((batch_size, n)) for n in neurons]
    new_spikes = [jnp.zeros((batch_size, n)) for n in neurons]
    traces = [jnp.zeros((batch_size, n)) for n in neurons]

    latent_spikes = [jnp.zeros((batch_size, n)) for n in neurons]

    output_voltage = jnp.zeros((batch_size, A[0].shape[0]))
    output_spikes = jnp.zeros((batch_size, A[0].shape[0]))
    output_trace = jnp.zeros((batch_size, A[0].shape[0]))
    output_spikes_count = jnp.zeros((batch_size, A[0].shape[0]))

    thrs, output_threshold = thresholds

    if plasticity:
        Yb_type = jnp.concat((jnp.ones((batch_size//2)), jnp.zeros((batch_size//2))))
    else:
        Yb_type = jnp.ones((batch_size))

    total_goodness = jnp.zeros((batch_size))

    input_spikes = jax.random.bernoulli(subkeys[1], Xb, shape=(T, batch_size, in_dim)).astype(jnp.float32)


    for t in range(T):
        #key, subkey = jax.random.split(key)
        #input_spikes = jax.random.bernoulli(subkey, Xb, shape=(batch_size, in_dim)).astype(jnp.float32)

        for l in range(num_layers):
            
            if l == 0:
                w_spikes = input_spikes[t]
            else:
                w_spikes = spikes[l-1]
            
            if l == num_layers-1: #Final layer
                _V = jnp.zeros_like(W[l])
                v_spikes = jnp.zeros_like(spikes[l-1])
                _v_opt = 0
            else:
                _V = V[l]
                v_spikes = spikes[l+1]
                _v_opt = v_opt[l]


            voltage_out, spikes_out, threshold_out, trace_out, dW, dV, dM, dB, _w_opt, _v_opt, _m_opt, _b_opt = _simulate([W[l], _V, M[l], B[l]], [w_spikes, v_spikes, spikes[l], Yb], voltages[l], thrs[l], traces[l], [w_opt[l], _v_opt, m_opt[l], b_opt[l]], dt, Yb_type, l != num_layers-1, plasticity)
            
            if record_latent:
                latent_spikes[l] += spikes_out

            voltages[l] = voltage_out
            new_spikes[l] = spikes_out
            thrs[l] = threshold_out
            traces[l] = trace_out
            
            m_opt[l] = _m_opt
            M[l] = dM
            w_opt[l] = _w_opt
            W[l] = dW

            if l != num_layers-1:
                v_opt[l] = _v_opt
                V[l] = dV

            b_opt[l] = _b_opt
            B[l] = dB
            
            #if l != 0:
            total_goodness += jnp.sum(jnp.square(traces[l]), axis=1)


        output_voltage, output_spikes, output_threshold, output_trace, hebbian_error = _classifier_simulate(A, spikes, output_voltage, output_threshold, output_trace, Yb, dt, Yb_type)

        output_spikes_count += output_spikes

        if plasticity:
            for l in range(num_layers):
                scaled_update = _hebbian_update(hebbian_error, spikes[l]) #* (1 - jnp.abs(A[l]))
                a_update, _temp = adam_update(scaled_update, a_opt[l])
                a_opt[l] = _temp
                A[l] = jnp.clip(A[l] - a_update, -1, 1)
        
        for l in range(num_layers):
            spikes[l] = new_spikes[l]

    if record_latent:
        return [W, V, M, A, B], [w_opt, v_opt, m_opt, a_opt, b_opt], jax.nn.softmax(output_spikes_count), output_spikes_count, total_goodness/T, jnp.concatenate(latent_spikes, axis=-1)

    return [W, V, M, A, B], [w_opt, v_opt, m_opt, a_opt, b_opt], jax.nn.softmax(output_spikes_count), output_spikes_count, total_goodness/T



@partial(jax.jit, static_argnames=["v_exists", "plasticity"])
def _simulate(weights, spikes, voltages, thresholds, traces, optim, dt, Yb_type, v_exists, plasticity):
    excitation = config.excitatory_resistance
    inhibition = config.inhibitory_resistance
    tau_m = config.tau_m
    lambda_v = config.lambda_v
    tau_tr = config.tau_tr
    gamma = config.gamma #Unused

    spikes_below, spikes_above, spikes_parallel, target_spikes = spikes

    W, V, M, B = weights

    current = excitation * (jnp.matmul(W, jnp.transpose(spikes_below)) + jnp.matmul(V, jnp.transpose(spikes_above)) + jnp.matmul(B, jnp.transpose(target_spikes))) - inhibition * jnp.matmul(M * (1 - jnp.identity(M.shape[0])), jnp.transpose(spikes_parallel))
    #current.shape -> (neurons l, batch_size)

    voltage_out = voltages + (dt / tau_m) * (jnp.transpose(current) - voltages)

    #spikes_out = (voltage_out > jnp.tile(jnp.transpose(jnp.array([thresholds])), (1, voltage_out.shape[1]))).astype(jnp.float32)
    spikes_out = (voltage_out > thresholds).astype(jnp.float32)

    voltage_out *= (1 - spikes_out)

    threshold_out = jnp.maximum(thresholds + lambda_v * (jnp.sum(spikes_out, axis=1, keepdims=True) - 1), 0.025)

    #trace_out = traces + (dt / tau_tr) * (gamma * spikes_out - traces)
    trace_out = (1 - dt / tau_tr) * traces * (1 - spikes_out) + spikes_out

    w_opt, v_opt, m_opt, b_opt = optim

    if plasticity:
        match config.training_type:
            case "standard": delta = jax.grad(_contrastive_loss)(trace_out, Yb_type)
            case "symba": delta = jax.grad(_symba_loss)(trace_out)

        dM = _delta_update(delta, spikes_parallel, inhibition)
        M_decay = _decay(spikes_parallel, spikes_out)

        dW = _delta_update(delta, spikes_below, excitation)
        W_decay = _decay(spikes_below, spikes_out)

        if v_exists:
            dV = _delta_update(delta, spikes_above, excitation)
            V_decay = _decay(spikes_above, spikes_out)
        else:
            dV = 0
            V_decay = 0
            
        dB = _delta_update(delta, target_spikes, excitation)
        B_decay = _decay(target_spikes, spikes_out)

        dM, m_opt = adam_update(dM, m_opt)
        dM = jnp.clip(M - dM - M_decay, 0, 1) #* (1 - jnp.identity(M.shape[0]))

        dW, w_opt = adam_update(dW, w_opt)
        dW = jnp.clip(W - dW - W_decay, -1, 1)

        if v_exists:
            dV, v_opt = adam_update(dV, v_opt)
            dV = jnp.clip(V - dV - V_decay, -1, 1)

        dB, b_opt = adam_update(dB, b_opt)
        dB = jnp.clip(B - dB - B_decay, -1, 1)

        return voltage_out, spikes_out, threshold_out, trace_out, dW, dV, dM, dB, w_opt, v_opt, m_opt, b_opt
    
    return voltage_out, spikes_out, threshold_out, trace_out, W, V, M, B, w_opt, v_opt, m_opt, b_opt



@jax.jit
def _delta_update(delta, pre_synaptic_spikes, resistance):
    du = jnp.transpose(resistance * jnp.matmul(jnp.transpose(pre_synaptic_spikes), delta))
    return du

@jax.jit
def _decay(pre_synaptic_spikes, post_synaptic_spikes):
    return jnp.transpose((config.lambda_d) * jnp.matmul(jnp.transpose(1 - pre_synaptic_spikes), post_synaptic_spikes))


@jax.jit
def _hebbian_update(error, spikes):
    excitation = config.excitatory_resistance
    return excitation * jnp.matmul(jnp.transpose(error), spikes)


@jax.jit
def _classifier_simulate(weights, spikes, output_voltage, output_threshold, output_trace, Yb, dt, Yb_type):
    excitation = config.excitatory_resistance
    tau_m = config.tau_m
    lambda_v = config.lambda_v
    tau_tr = config.tau_tr
    
    sum = jnp.zeros((spikes[0].shape[0], weights[0].shape[0])) #(batch_size, output_size)
    for l in range(len(weights)):
        sum += jnp.matmul(spikes[l], jnp.transpose(weights[l]))
    
    #sum = jnp.matmul(spikes[1], jnp.transpose(weights[1])) #-> worse

    #current = excitation * (jnp.sum(map(lambda w, s: jnp.matmul(w, jnp.transpose(s)), weights, spikes), axis=1))
    current = sum #* excitation

    voltage_out = output_voltage + (dt / tau_m) * (current - output_voltage)

    #spikes_out = (voltage_out > jnp.tile(jnp.transpose(jnp.array([output_threshold])), (1, voltage_out.shape[1]))).astype(jnp.float32)
    spikes_out = (voltage_out > output_threshold).astype(jnp.float32)

    voltage_out *= (1 - spikes_out)

    threshold_out = jnp.maximum(output_threshold + lambda_v * (jnp.sum(spikes_out, axis=1, keepdims=True) - 1), 0.025)

    trace_out = (1 - dt / tau_tr) * output_trace * (1 - spikes_out) + spikes_out
    
    #hebbian_error = spikes_out - Yb
    hebbian_error = trace_out - Yb 

    mod = jnp.expand_dims(Yb_type, axis=1)
    scaled_error = hebbian_error * mod * (1/(jnp.sum(mod))) * 2

    return voltage_out, spikes_out, threshold_out, trace_out, scaled_error
