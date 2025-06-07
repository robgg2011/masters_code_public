import cartpole_config as config

import jax
import jax.numpy as jnp
from jax import random

from functools import partial

import torch
import torch.nn as nn

import numpy as np
        


def csdp_init_agent(random_key):
    hidden = config.neurons.copy()
    hidden.insert(0, config.input_size)

    key, *subkeys= random.split(random_key, len(hidden)*5)
        
    W = [random.uniform(subkeys[i-1],(hidden[i], hidden[i-1]), minval=-1) for i in range(1, len(hidden))]  #matrix of weights TO each layer, from layer below. self.W[l][i][j] -> the weight from neuron j in layer l-1 to neuron i in layer l
    V = [random.uniform(subkeys[i-1 + len(hidden)],(hidden[i], hidden[i+1]), minval=-1) for i in range(1, len(hidden)-1)] 
    M = [random.uniform(subkeys[i-1 + len(hidden)*2 - 1],(hidden[i], hidden[i]), minval= 0) for i in range(1, len(hidden))]

    A = [random.uniform(subkeys[i-1+ len(hidden)*3 - 1],(config.num_classes, hidden[i]), minval=-1) for i in range(1, len(hidden))]
    B = [random.uniform(subkeys[i-1+ len(hidden)*4 - 1],(hidden[i], config.num_classes), minval=-1) for i in range(1, len(hidden))]

    # Eligibility #
    w_state = [jnp.zeros_like(l) for l in W]
    v_state = [jnp.zeros_like(l) for l in V]
    m_state = [jnp.zeros_like(l) for l in M]
    #a_state = [jnp.zeros_like(l) for l in A]
    b_state = [jnp.zeros_like(l) for l in B]

    num_layers = len(config.neurons)

    maxthr = config.SPIKE_THRESHOLD+config.SPIKE_THRESHOLD_JITTER
    minthr = config.SPIKE_THRESHOLD-config.SPIKE_THRESHOLD_JITTER

    key, *thrkeys = jax.random.split(key, num_layers+1)
    thresholds = [jax.random.uniform(thrkeys[i], (1, config.neurons[i]), minval=minthr, maxval=maxthr) for i in range(num_layers)] 

    key, out_thrkey = jax.random.split(key)
    output_threshold = jax.random.uniform(out_thrkey, (1, config.num_classes), minval=minthr, maxval=maxthr)

    return [[W, V, M, A, B], [w_state, v_state, m_state, b_state]], [thresholds, output_threshold]





def csdp_reset_el_trace(weights):
    W, V, M, A, B = weights
    w_state = [jnp.zeros_like(l) for l in W]
    v_state = [jnp.zeros_like(l) for l in V]
    m_state = [jnp.zeros_like(l) for l in M]
    #a_state = [jnp.zeros_like(l) for l in A]
    b_state = [jnp.zeros_like(l) for l in B]
    return [w_state, v_state, m_state, b_state]


@jax.jit
def generate_negative_data(Xb, Yb, random_key):
    Xb_neg = Xb

    Yb_neg = jax.random.uniform(random_key, Yb.shape, minval=0., maxval=1.) * (1. - Yb)
    Yb_neg = jax.nn.one_hot(jnp.argmax(Yb_neg, axis=1), num_classes=Yb.shape[1], dtype=jnp.float32)

    return Xb, Yb, Yb_neg, Xb_neg


@jax.jit
def _contrastive_loss(z, y_type):
    delta = jnp.sum(jnp.square(z), axis=1) - config.goodness_threshold

    out = jnp.maximum(delta, 0) - delta * (y_type) + jnp.log(1. + jnp.exp(-jnp.abs(delta)))

    return jnp.mean(out)


@jax.jit
def _symba_loss(z):
    z_pos = z[:config.batch_size]
    z_neg = z[config.batch_size:]
    
    delta_symba = jnp.sum(jnp.square(z_pos), axis=1) - jnp.sum(jnp.square(z_neg), axis=1)

    #out = jnp.log(1. + jnp.exp(- config.alpha * delta_symba)) # This is unstable
    
    out = -jnp.minimum(delta_symba, 0) + jnp.log(1. + jnp.exp(- config.alpha * jnp.abs(delta_symba)))

    return jnp.mean(out)



@partial(jax.jit, static_argnames=["plasticity", "y_type"])
def agent_csdp_process(Xb, Yb, weights, thresholds, eligibility, step, y_type, random_key, plasticity=True, td_delta=None, *, discount=config.discount, actor_decay=config.actor_decay, actor_lr= config.actor_lr, goodness_threshold=config.goodness_threshold, alpha=config.alpha, lambda_d=config.lambda_d):
    key, *subkeys = jax.random.split(random_key, 3)
    del random_key

    excitation = config.excitatory_resistance
    inhibition = config.inhibitory_resistance
    tau_m = config.tau_m
    lambda_v = config.lambda_v
    tau_tr = config.tau_tr
    
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
    w_el, v_el, m_el, b_el = eligibility

    w_el_0 = [jnp.copy(e) / T for e in w_el]
    v_el_0 = [jnp.copy(e) / T for e in v_el]
    m_el_0 = [jnp.copy(e) / T for e in m_el]
    b_el_0 = [jnp.copy(e) / T for e in b_el]

    batch_size = Xb.shape[0]
    num_layers = len(W)
    in_dim = Xb.shape[1]
    neurons = [w.shape[0] for w in W]

    voltages = [jnp.zeros((batch_size, n)) for n in neurons]
    spikes = [jnp.zeros((batch_size, n)) for n in neurons]
    new_spikes = [jnp.zeros((batch_size, n)) for n in neurons]
    traces = [jnp.zeros((batch_size, n)) for n in neurons]
    
    output_voltage = jnp.zeros((batch_size, A[0].shape[0]))
    output_spikes = jnp.zeros((batch_size, A[0].shape[0]))
    output_trace = jnp.zeros((batch_size, A[0].shape[0]))
    output_spikes_count = jnp.zeros((batch_size, A[0].shape[0]))

    thrs, output_threshold = thresholds

    if plasticity:
        if y_type == 1:
            Yb_type = jnp.concat((jnp.ones((batch_size//2)), jnp.zeros((batch_size//2))))
        else:
            Yb_type = jnp.concat((jnp.zeros((batch_size//2)), jnp.ones((batch_size//2))))
    else:
        Yb_type = jnp.ones((batch_size))

    total_goodness = jnp.zeros((batch_size))

    input_spikes = jax.random.bernoulli(subkeys[1], Xb, shape=(T, batch_size, in_dim)).astype(jnp.float32)


    for t in range(T):
        for l in range(num_layers):
            
            if l == 0:
                w_spikes = input_spikes[t]
            else:
                w_spikes = spikes[l-1]
            
            if l == num_layers-1: #Final layer
                _V = jnp.zeros_like(W[l])
                v_spikes = jnp.zeros_like(spikes[l-1])
            else:
                _V = V[l]
                v_spikes = spikes[l+1]


            current = excitation * (jnp.matmul(W[l], jnp.transpose(w_spikes)) + jnp.matmul(_V, jnp.transpose(v_spikes)) + jnp.matmul(B[l], jnp.transpose(Yb))) - inhibition * jnp.matmul(M[l] * (1 - jnp.identity(M[l].shape[0])), jnp.transpose(spikes[l]))
            #current.shape -> (neurons l, batch_size)

            voltages[l] = voltages[l] + (dt / tau_m) * (jnp.transpose(current) - voltages[l])

            new_spikes[l] = (voltages[l] > thrs[l]).astype(jnp.float32)

            voltages[l] *= (1 - new_spikes[l])

            thrs[l] = jnp.maximum(thrs[l] + lambda_v * (jnp.sum(new_spikes[l], axis=1, keepdims=True) - 1), 0.025)

            traces[l] = (1 - dt / tau_tr) * traces[l] * (1 - new_spikes[l]) + new_spikes[l]

            if plasticity:
                match config.training_type:
                    case "standard": delta = jax.value_and_grad(_contrastive_loss)(traces[l], Yb_type)
                    case "symba": delta = jax.value_and_grad(_symba_loss)(traces[l])

                dM = (actor_decay**step) * _delta_update(delta[1], spikes[l], inhibition) / (delta[0] + 1e-8)
                m_el[l] += dM
                M_decay = _decay(spikes[l], new_spikes[l], lambda_d)

                dW = (actor_decay**step) *_delta_update(delta[1], w_spikes, excitation) / (delta[0] + 1e-8)
                w_el[l] += dW
                W_decay = _decay(w_spikes, new_spikes[l], lambda_d)

                if l != num_layers-1:
                    dV = (actor_decay**step) * _delta_update(delta[1], v_spikes, excitation) / (delta[0] + 1e-8)
                    v_el[l] += dV
                    V_decay = _decay(v_spikes, new_spikes[l], lambda_d)
                else:
                    dV = 0
                    V_decay = 0
                    
                dB = (actor_decay**step) * _delta_update(delta[1], Yb, excitation) / (delta[0] + 1e-8)
                b_el[l] += dB
                B_decay = _decay(Yb, new_spikes[l], lambda_d)

                dM = jnp.clip(M[l] - actor_lr * abs(td_delta) * (m_el_0[l] + dM) - M_decay, 0, 1)

                dW = jnp.clip(W[l] - actor_lr * abs(td_delta) * (w_el_0[l] + dW) - W_decay, -1, 1)

                if l != num_layers-1:
                    dV = jnp.clip(V[l] - actor_lr * abs(td_delta) * (v_el_0[l] + dV) - V_decay, -1, 1)

                dB = jnp.clip(B[l] - actor_lr * abs(td_delta) * (b_el_0[l] + dB) - B_decay, -1, 1)
            
            #if l != 0:
            total_goodness += jnp.sum(jnp.square(traces[l]), axis=1)


        output_voltage, output_spikes, output_threshold, output_trace, hebbian_error = _classifier_simulate(A, spikes, output_voltage, output_threshold, output_trace, Yb, dt, Yb_type)

        output_spikes_count += output_spikes

        if plasticity:
            for l in range(num_layers):
                a_update = _hebbian_update(hebbian_error, spikes[l])
                A[l] = jnp.clip(A[l] - actor_lr * a_update, -1, 1)
        
        for l in range(num_layers):
            spikes[l] = new_spikes[l]


    return [W, V, M, A, B], [w_el, v_el, m_el, b_el], jax.nn.softmax(output_spikes_count), output_spikes_count, total_goodness/T



@jax.jit
def _delta_update(delta, pre_synaptic_spikes, resistance):
    du = jnp.transpose(resistance * jnp.matmul(jnp.transpose(pre_synaptic_spikes), delta))
    return du

@jax.jit
def _decay(pre_synaptic_spikes, post_synaptic_spikes, lambda_d):
    return jnp.transpose((lambda_d) * jnp.matmul(jnp.transpose(1 - pre_synaptic_spikes), post_synaptic_spikes))


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



@partial(jax.jit, static_argnames=["plasticity", "y_type"])
def classifier_only_agent_csdp_process(Xb, Yb, weights, thresholds, eligibility, step, y_type, random_key, plasticity=True, td_delta=None, *, discount=config.discount, actor_decay=config.actor_decay, actor_lr= config.actor_lr, goodness_threshold=config.goodness_threshold, alpha=config.alpha, lambda_d=config.lambda_d):
    key, *subkeys = jax.random.split(random_key, 3)
    del random_key

    excitation = config.excitatory_resistance
    inhibition = config.inhibitory_resistance
    tau_m = config.tau_m
    lambda_v = config.lambda_v
    tau_tr = config.tau_tr
    
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
    w_el, v_el, m_el, b_el = eligibility

    w_el_0 = [jnp.copy(e) / T for e in w_el]
    v_el_0 = [jnp.copy(e) / T for e in v_el]
    m_el_0 = [jnp.copy(e) / T for e in m_el]
    b_el_0 = [jnp.copy(e) / T for e in b_el]

    batch_size = Xb.shape[0]
    num_layers = len(W)
    in_dim = Xb.shape[1]
    neurons = [w.shape[0] for w in W]

    voltages = [jnp.zeros((batch_size, n)) for n in neurons]
    spikes = [jnp.zeros((batch_size, n)) for n in neurons]
    new_spikes = [jnp.zeros((batch_size, n)) for n in neurons]
    traces = [jnp.zeros((batch_size, n)) for n in neurons]
    
    output_voltage = jnp.zeros((batch_size, A[0].shape[0]))
    output_spikes = jnp.zeros((batch_size, A[0].shape[0]))
    output_trace = jnp.zeros((batch_size, A[0].shape[0]))
    output_spikes_count = jnp.zeros((batch_size, A[0].shape[0]))

    thrs, output_threshold = thresholds

    if plasticity:
        if y_type == 1:
            Yb_type = jnp.concat((jnp.ones((batch_size//2)), jnp.zeros((batch_size//2))))
        else:
            Yb_type = jnp.concat((jnp.zeros((batch_size//2)), jnp.ones((batch_size//2))))
    else:
        Yb_type = jnp.ones((batch_size))

    total_goodness = jnp.zeros((batch_size))

    input_spikes = jax.random.bernoulli(subkeys[1], Xb, shape=(T, batch_size, in_dim)).astype(jnp.float32)


    for t in range(T):
        for l in range(num_layers):
            
            if l == 0:
                w_spikes = input_spikes[t]
            else:
                w_spikes = spikes[l-1]
            
            if l == num_layers-1: #Final layer
                _V = jnp.zeros_like(W[l])
                v_spikes = jnp.zeros_like(spikes[l-1])
            else:
                _V = V[l]
                v_spikes = spikes[l+1]


            current = excitation * (jnp.matmul(W[l], jnp.transpose(w_spikes)) + jnp.matmul(_V, jnp.transpose(v_spikes)) + jnp.matmul(B[l], jnp.transpose(Yb))) - inhibition * jnp.matmul(M[l] * (1 - jnp.identity(M[l].shape[0])), jnp.transpose(spikes[l]))
            #current.shape -> (neurons l, batch_size)

            voltages[l] = voltages[l] + (dt / tau_m) * (jnp.transpose(current) - voltages[l])

            new_spikes[l] = (voltages[l] > thrs[l]).astype(jnp.float32)

            voltages[l] *= (1 - new_spikes[l])

            thrs[l] = jnp.maximum(thrs[l] + lambda_v * (jnp.sum(new_spikes[l], axis=1, keepdims=True) - 1), 0.025)

            traces[l] = (1 - dt / tau_tr) * traces[l] * (1 - new_spikes[l]) + new_spikes[l]

            if plasticity:
                match config.training_type:
                    case "standard": delta = jax.value_and_grad(_contrastive_loss)(traces[l], Yb_type)
                    case "symba": delta = jax.value_and_grad(_symba_loss)(traces[l])

                dM = (actor_decay**step) * _delta_update(delta[1], spikes[l], inhibition) / (delta[0] + 1e-8)
                m_el[l] += dM
                M_decay = _decay(spikes[l], new_spikes[l], lambda_d)

                dW = (actor_decay**step) *_delta_update(delta[1], w_spikes, excitation) / (delta[0] + 1e-8)
                w_el[l] += dW
                W_decay = _decay(w_spikes, new_spikes[l], lambda_d)

                if l != num_layers-1:
                    dV = (actor_decay**step) * _delta_update(delta[1], v_spikes, excitation) / (delta[0] + 1e-8)
                    v_el[l] += dV
                    V_decay = _decay(v_spikes, new_spikes[l], lambda_d)
                else:
                    dV = 0
                    V_decay = 0
                    
                dB = (actor_decay**step) * _delta_update(delta[1], Yb, excitation) / (delta[0] + 1e-8)
                b_el[l] += dB
                B_decay = _decay(Yb, new_spikes[l], lambda_d)

                dM = jnp.clip(M[l] - actor_lr * abs(td_delta) * (m_el_0[l] + dM) - M_decay, 0, 1)

                dW = jnp.clip(W[l] - actor_lr * abs(td_delta) * (w_el_0[l] + dW) - W_decay, -1, 1)

                if l != num_layers-1:
                    dV = jnp.clip(V[l] - actor_lr * abs(td_delta) * (v_el_0[l] + dV) - V_decay, -1, 1)

                dB = jnp.clip(B[l] - actor_lr * abs(td_delta) * (b_el_0[l] + dB) - B_decay, -1, 1)
            
            #if l != 0:
            total_goodness += jnp.sum(jnp.square(traces[l]), axis=1)


        output_voltage, output_spikes, output_threshold, output_trace, hebbian_error = _classifier_simulate(A, spikes, output_voltage, output_threshold, output_trace, Yb, dt, Yb_type)

        output_spikes_count += output_spikes

        if plasticity:
            for l in range(num_layers):
                a_update = _hebbian_update(hebbian_error, spikes[l])
                A[l] = jnp.clip(A[l] - actor_lr * a_update, -1, 1)
        
        for l in range(num_layers):
            spikes[l] = new_spikes[l]


    return [W, V, M, A, B], [w_el, v_el, m_el, b_el], jax.nn.softmax(output_spikes_count), output_spikes_count, total_goodness/T



def classifier_only_csdp_init_agent(random_key):
    hidden = config.neurons.copy()
    hidden.insert(0, config.input_size)

    key, *subkeys= random.split(random_key, len(hidden)*5)
        
    W = [random.uniform(subkeys[i-1],(hidden[i], hidden[i-1]), minval=-1) for i in range(1, len(hidden))]  #matrix of weights TO each layer, from layer below. self.W[l][i][j] -> the weight from neuron j in layer l-1 to neuron i in layer l
    V = [random.uniform(subkeys[i-1 + len(hidden)],(hidden[i], hidden[i+1]), minval=-1) for i in range(1, len(hidden)-1)] 
    M = [random.uniform(subkeys[i-1 + len(hidden)*2 - 1],(hidden[i], hidden[i]), minval= 0) for i in range(1, len(hidden))]

    A = [random.uniform(subkeys[len(hidden)*3 - 1],(config.num_classes, hidden[0]), minval=-1)]
    B = [random.uniform(subkeys[i-1+ len(hidden)*4 - 1],(hidden[i], config.num_classes), minval=-1) for i in range(1, len(hidden))]

    # Eligibility #
    w_state = [jnp.zeros_like(l) for l in W]
    v_state = [jnp.zeros_like(l) for l in V]
    m_state = [jnp.zeros_like(l) for l in M]
    #a_state = [jnp.zeros_like(l) for l in A]
    b_state = [jnp.zeros_like(l) for l in B]

    num_layers = len(config.neurons)

    maxthr = config.SPIKE_THRESHOLD+config.SPIKE_THRESHOLD_JITTER
    minthr = config.SPIKE_THRESHOLD-config.SPIKE_THRESHOLD_JITTER

    key, *thrkeys = jax.random.split(key, num_layers+1)
    thresholds = [jax.random.uniform(thrkeys[i], (1, config.neurons[i]), minval=minthr, maxval=maxthr) for i in range(num_layers)] 

    key, out_thrkey = jax.random.split(key)
    output_threshold = jax.random.uniform(out_thrkey, (1, config.num_classes), minval=minthr, maxval=maxthr)

    return [[W, V, M, A, B], [w_state, v_state, m_state, b_state]], [thresholds, output_threshold]