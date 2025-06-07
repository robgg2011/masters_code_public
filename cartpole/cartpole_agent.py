import cartpole_config as config

import torch.nn as nn
import numpy as np
import gymnasium as gym

import torch

from custom_torch import LogitClip, weight_init_normal, weight_init_uniform

from ff_agent_functional_library import ff_init_agent, agent_ff_process
from csdp_agent_functional_library import csdp_init_agent, agent_csdp_process, csdp_reset_el_trace

import jax.nn
import jax.numpy as jnp


class Cartpole_Agent():
    def __init__(self, env: gym.Env, hidden, discount=config.discount, C=config.C, actor_decay = config.actor_decay, actor_lr = config.actor_lr, critic_decay =  config.critic_decay, critic_lr= config.critic_lr, goodness_threshold=config.goodness_threshold, alpha=config.alpha, lambda_d=config.lambda_d, random_key = None):
        self.epsilon_greed = 1

        self.discount = discount

        self.actor_decay =actor_decay
        self.actor_lr = actor_lr
        self.critic_decay =critic_decay
        self.critic_lr = critic_lr


        ### Critic ##

        self.critic = nn.Sequential(
            nn.Linear(np.prod(env.observation_space.shape, 0), hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )
        self.critic.type(torch.double)

        self.critic.apply(weight_init_normal)
        
        self.critic_eligibility = [torch.zeros_like(p) for p in self.critic.parameters()]
        

        ### Actor ###
        self.actor_type = config.actor_type

        if self.actor_type == "standard":
            self.actor = nn.Sequential(
                nn.Linear(np.prod(env.observation_space.shape, 0), hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, env.action_space.n),
                LogitClip(C),
                nn.Softmax()
            )
            self.actor.type(torch.double)
            
            #self.actor.apply(weight_init_uniform)

            self.actor_eligibility = [torch.zeros_like(p) for p in self.actor.parameters()]
        elif self.actor_type == "ff":
            self.actor_params = ff_init_agent(random_key)
            self.goodness_threshold = goodness_threshold
            self.alpha=alpha
        elif self.actor_type == "csdp":
            self.actor_params, self.base_thresholds = csdp_init_agent(random_key)
            self.goodness_threshold = goodness_threshold
            self.alpha=alpha

            self.lambda_d = lambda_d
        else:
            raise ValueError("Invalid actor type for CartPole Agent")
        

    def episode_reset(self):
        #for t in self.actor_eligibility:
        #   t.zero_()
        if config.actor_type == "standard":
            self.actor_eligibility = [torch.zeros_like(p) for p in self.actor_eligibility]
        elif self.actor_type == "ff":
            weights, actor_eligibility = self.actor_params
            self.actor_params = [weights, [jnp.zeros_like(weights[0][l]) for l in range(len(weights[0]))]]
        elif self.actor_type == "csdp":
            weights, actor_eligibility = self.actor_params
            self.actor_params = [weights, csdp_reset_el_trace(weights)]
            

        self.critic_eligibility = [torch.zeros_like(p) for p in self.critic_eligibility]
        

    def get_action(self, obs, random_key = None, testing = False):
        #if np.random.uniform(0, 1) < self.epsilon_greed:
        #    if self.epsilon_greed > 0.1: self.epsilon_greed -= 0.01
        #    return np.random.choice(5)
        if self.actor_type == "standard":
            x = torch.flatten(torch.tensor(obs, dtype=float))

            with torch.no_grad():
                prob_dist = self.actor(x)

            return np.random.choice(range(len(prob_dist)), p=prob_dist)


        ### Actor type == else ###

        if self.actor_type == "ff":
            if np.random.uniform(0, 1) < self.epsilon_greed and not testing:
                if self.epsilon_greed > 0.05: self.epsilon_greed -= 0.01
                #print("greed is good")
                return np.random.choice(2)

            weights, actor_eligibility = self.actor_params

            x = jnp.asarray(obs)
            x = x / x.max()
            
            """_goodnessses = []
            
            debug = ""
            key = random_key
            best_class = (-1, -jnp.inf)
            for i in range(config.num_classes):
                key, subkey = jax.random.split(key)
                _, _, _, goodness = agent_ff_process(jnp.array((x,)), jax.nn.one_hot((i,), config.num_classes), weights, actor_eligibility, 0, 1, random_key, plasticity=False, goodness_threshold = self.goodness_threshold, alpha=self.alpha)

                _goodnessses.append(goodness)

                #debug += f"Action {i}: {goodness} / "
                if best_class[1] < goodness:
                    best_class = (i, goodness)

            #print(debug)

            return best_class[0]"""

            _, _, prob_dist, _ = agent_ff_process(jnp.array((x,)), jnp.zeros((1, config.num_classes)), weights, actor_eligibility, 0, 1, random_key, plasticity=False, goodness_threshold = self.goodness_threshold, alpha=self.alpha)

            p = np.array(prob_dist[0])
            p[-1] = max(0, 1 - np.sum(p[0:-1]))

            #print(prob_dist)
            return np.random.choice(np.arange(len(p)), p=p)
        

        if self.actor_type == "csdp":
            if np.random.uniform(0, 1) < self.epsilon_greed:
                if self.epsilon_greed > 0.05: self.epsilon_greed -= 0.01
                #print("greed is good")
                return np.random.choice(2)
            

            weights, actor_eligibility = self.actor_params

            thr = [[t+0 for t in self.base_thresholds[0]], self.base_thresholds[1]+0]

            x = jnp.asarray(obs)
            x = x / x.max()
            
            debug = ""
            key = random_key

            """_, _, prob_dist, _, _ = agent_csdp_process(jnp.array((x,)), jnp.zeros((1, config.num_classes)), weights, thr, actor_eligibility, 0, 1, random_key, plasticity=False, goodness_threshold=self.goodness_threshold, alpha=self.alpha, lambda_d = self.lambda_d)

            #print(f"Actions :{prob_dist}")
            p = np.array(prob_dist[0])
            p[-1] = max(0, 1 - np.sum(p[0:-1]))
            return np.random.choice(np.arange(len(p)), p=p)"""


            key = random_key
            best_class = (-1, -jnp.inf)
            for i in range(config.num_classes):
                key, subkey = jax.random.split(key)
                _, _, _, _, goodness = agent_csdp_process(jnp.array((x,)), jax.nn.one_hot((i,), config.num_classes), weights, thr, actor_eligibility, 0, 1, random_key, plasticity=False, goodness_threshold=self.goodness_threshold, alpha=self.alpha, lambda_d = self.lambda_d)

                #debug += f"Action {i}: {goodness} / "
                if best_class[1] < goodness:
                    best_class = (i, goodness)

            #print(debug)

            return best_class[0]


        
            
        
    def update(self, obs, action, reward, next_obs, step, done, random_key=None):
        """ Update policy network.

        :param episode_data: batch of input observations (state), action, reward, and next observations
        """
        
        obs = torch.flatten(torch.tensor(obs, dtype=float))
        next_obs = torch.flatten(torch.tensor(next_obs, dtype=float))

        #w_c = 0
        #for p in self.critic.parameters():
            #print(p.max())
            #w_c += torch.sum(p**2)

        current_value = self.critic(obs)
        critic_grad = current_value
        critic_grad.backward()

        if self.actor_type == "standard":
            prob_dist = self.actor(obs)
            policy_out = torch.log(prob_dist[action])
            policy_out.backward()

        elif self.actor_type == "ff" or self.actor_type == "csdp":
            weights, actor_eligibility = self.actor_params
        
        with torch.no_grad():
            if not done:
                new_value = self.critic(next_obs)
            else:
                new_value = 0

            delta = reward + self.discount * new_value - current_value

            #print(f"CRITIC : {current_value}")
            #print(f"NEW STATE : {new_value}")
            #print(f"DELTA : {delta}")
            #print(f"GRADIENT -> {self.critic.weight.grad}")
            
            if self.actor_type == "standard":
                i = 0
                for p in self.actor.parameters():
                    self.actor_eligibility[i] = self.discount * self.actor_decay * self.actor_eligibility[i] + (self.discount ** step) * p.grad
                    new_val = p + self.actor_lr * delta * self.actor_eligibility[i]
                    p.copy_(new_val)

                    i += 1

            elif self.actor_type == "ff":
                x = jnp.asarray(obs)
                x = x / x.max()
                y = jax.nn.one_hot(action, num_classes=config.num_classes, dtype=jnp.float32)
                weights, actor_eligibility, _, _ = agent_ff_process(jnp.array((x,)), jnp.asarray((y,)), weights, actor_eligibility, step, int(delta > 0), random_key, plasticity=True, td_delta = jnp.array(delta), goodness_threshold=self.goodness_threshold, actor_decay= self.actor_decay, actor_lr=self.actor_lr)
                self.actor_params = [weights, actor_eligibility]

            elif self.actor_type == "csdp":
                x = jnp.asarray(obs)
                x = x / x.max()
                y = jax.nn.one_hot(action, num_classes=config.num_classes, dtype=jnp.float32)

                thr = [[t+0 for t in self.base_thresholds[0]], self.base_thresholds[1]+0]

                weights, actor_eligibility, prob_dist, _, _ = agent_csdp_process(jnp.array((x,)), jnp.asarray((y,)), weights, thr, actor_eligibility, step, int(delta > 0), random_key, plasticity=True, td_delta = jnp.array(delta), goodness_threshold=self.goodness_threshold, actor_decay= self.actor_decay, actor_lr=self.actor_lr, lambda_d = self.lambda_d)
                self.actor_params = [weights, actor_eligibility]
            
            j = 0
            for p in self.critic.parameters():
                #print(f"GRADIENT -> {p.grad}")
                #print(f"Eligibility -> {self.critic_eligibility[j]}")

                self.critic_eligibility[j] = self.discount * self.critic_decay * self.critic_eligibility[j] + np.clip(p.grad, -1, 1) #gradient clipping
                new_val = p + self.critic_lr * delta * self.critic_eligibility[j]
                p.copy_(new_val)

                #print(f"New weight -> {new_val}")

                j += 1


        if self.actor_type == "standard": self.actor.zero_grad()
        self.critic.zero_grad()
        

        


