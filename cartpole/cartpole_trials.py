import cartpole_config as config

import jax
import gymnasium as gym
import numpy as np

from cartpole_agent import Cartpole_Agent

import matplotlib.pyplot as plt



def train(env, agent, key):
    
    score_queue = []
    for episode in range(config.n_train_episodes):
        agent.episode_reset()
        
        obs, info = env.reset()
        done = False

        score = 0
        step = 0
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
            
            score += reward

            obs = next_obs
            step += 1
        
        score_queue.append(score)

    return score_queue


def test(env, agent, key):
    score_queue = []
    for episode in range(config.n_test_episodes):
        agent.episode_reset()
        
        obs, info = env.reset()
        done = False

        score = 0
        while not done:
            # Basic Gym steps
            key, subkey = jax.random.split(key)
            action = agent.get_action(obs, key, testing=False)
            #action = agent.get_action(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated   

            obs = next_obs

            score += 1
        
        score -= 1
        score_queue.append(score)

    return score_queue


def main():
    trial_records = []

    score_record = []

    env = gym.make("CartPole-v1", render_mode="rgb_array", sutton_barto_reward=False)

    for trial in range(1, config.num_trials+1):
        key = jax.random.key(config.EXPERIMENTAL_SEEDS[trial-1])
        key, *subkeys = jax.random.split(key, 4)
        agent = Cartpole_Agent(env, config.hidden, random_key=subkeys[0])

        try:
            record = train(env, agent, subkeys[1])
            score = test(env, agent, subkeys[2])

            out = np.mean(score)

            print(f"Trial {trial} - Mean test score: {out}")

            score_record.append(score)
            trial_records.append(record)
        except:
            print(f"Trial {trial} - Mean test score: NaN -> record discarded")

        
    


    print(trial_records)

    score_queue = np.mean(trial_records, axis=0)
    print(score_queue)

    #Display training data
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))

    axs[0].plot(score_queue, label="Mean score (raw)")
    axs[0].plot(np.convolve(score_queue, np.ones(10), mode="valid") / 10, label="Mean score (10 episode average)")
    #axs[0].plot(score_queue)
    axs[0].set_title("Episode Rewards")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Reward")

    plt.legend()

    """axs[1].plot(np.convolve(score_queue, np.ones(100), mode="valid") / 100)
    #axs[1].plot(env.length_queue)
    axs[1].set_title("Episode Rewards 100")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Rewards 100")"""

    std = np.std(trial_records, axis=0)
    #print(std)
    
    plt.errorbar(np.arange(len(score_queue)), score_queue, yerr=[std, std], capsize=3, ecolor = "black")


    print(f"{config.actor_type} {config.training_type} Test: mean = {np.mean(score_record)} std = { np.std(score_record)} | lr = {config.actor_lr} lambda_d = {config.lambda_d} actor_decay = {config.actor_decay}")
    
    
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()


