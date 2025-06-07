import cartpole_config as config

import gymnasium as gym
import numpy as np

from tqdm import trange
from matplotlib import pyplot as plt

from cartpole_agent import Cartpole_Agent

import jax
import torch.autograd


torch.autograd.set_detect_anomaly(True)

def play_one_episode(env, agent, obs):
    done = False
    while not done:
        # Basic Gym steps
        action = agent.get_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        obs = next_obs

def play_one_episode(env, agent, obs, random_key):
    key = random_key
    
    done = False
    while not done:
        # Basic Gym steps
        key, subkey = jax.random.split(key)
        action = agent.get_action(obs, key)

        next_obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        obs = next_obs



def main():
    n_episodes = config.n_episodes

    env = gym.make("CartPole-v1", render_mode="rgb_array", sutton_barto_reward=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    key = jax.random.key(config.SEED)
    key, subkey = jax.random.split(key)

    agent = Cartpole_Agent(env, config.hidden, random_key = subkey)

    score_queue = []

    """wrapped = gym.wrappers.HumanRendering(env)
    obs, _ = wrapped.reset()
    play_one_episode(wrapped, agent, obs)"""
    

    for episode in trange(n_episodes):
        agent.episode_reset()
        
        obs, info = env.reset()
        done = False

        step = 0

        score = 0

        while not done:
            # Basic Gym steps
            key, subkey = jax.random.split(key)
            action = agent.get_action(obs, key)
            #action = agent.get_action(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            #reward -= 0.5 #time punishment

            key, subkey = jax.random.split(key)
            agent.update(obs, action, reward, next_obs, step, done, key)
            #agent.update(obs, action, reward, next_obs, step, done)
            

            obs = next_obs
            step += 1

            #score += reward
            score += 1

        score_queue.append(score)
    
    print(score_queue)


    #Display one game
    wrapped = gym.wrappers.HumanRendering(env)
    obs, _ = wrapped.reset()
    #play_one_episode(wrapped, agent, obs)
    key, subkey = jax.random.split(key)
    play_one_episode(wrapped, agent, obs, key)
    
    env.close()


    #Display training data
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))


    axs[0].plot(np.convolve(score_queue, np.ones(10), mode="valid") / 10)
    #axs[0].plot(score_queue)
    axs[0].set_title("Episode Rewards")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Reward")

    axs[1].plot(np.convolve(score_queue, np.ones(100), mode="valid") / 100)
    #axs[1].plot(env.length_queue)
    axs[1].set_title("Episode Rewards 100")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Rewards 100")

    axs[2].plot(score_queue)
    #axs[1].plot(env.length_queue)
    axs[2].set_title("Episode Rewards Raw")
    axs[2].set_xlabel("Episode")
    axs[2].set_ylabel("Rewards raw")
    """axs[1].plot(np.convolve(env.length_queue, np.ones(10)))
    #axs[1].plot(env.length_queue)
    axs[1].set_title("Episode Lengths")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Length")"""

    plt.tight_layout()
    plt.show()

    plt.savefig(f"cartpole_plot.png", bbox_inches='tight')







if __name__ == "__main__":
    main()