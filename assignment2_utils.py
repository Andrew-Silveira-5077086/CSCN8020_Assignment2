#!/usr/bin/python3
import numpy as np
import time
#import gym
import gymnasium as gym
from taxi_agent import Agent

#---------------------------
# Helper functions
#---------------------------

'''@brief Describes the environment actions, observation states, and reward range
'''
def describe_env(env: gym.Env):
    num_actions = env.action_space.n
    obs = env.observation_space
    num_obs = env.observation_space.n
    #reward_range = env.reward_range
    reward_range = (-float("inf"), float("inf"))
    action_desc = { 
        0: "Move south (down)",
        1: "Move north (up)",
        2: "Move east (right)",
        3: "Move west (left)",
        4: "Pickup passenger",
        5: "Drop off passenger"
    }
    print("Observation space: ", obs)
    print("Observation space size: ", num_obs)
    print("Reward Range: ", reward_range)
    
    print("Number of actions: ", num_actions)
    print("Action description: ", action_desc)
    return num_obs, num_actions


'''@brief Get the string description of the action
'''
def get_action_description(action):
    action_desc = { 
        0: "Move south (down)",
        1: "Move north (up)",
        2: "Move east (right)",
        3: "Move west (left)",
        4: "Pickup passenger",
        5: "Drop off passenger"
    }
    return action_desc[action]

'''@brief print full description of current observation
'''
def describe_obs(obs):
    obs_desc = {
        0: "Red",
        1: "Green",
        2: "Yellow",
        3: "Blue",
        4: "In taxi"
    }
    obs_dict = breakdown_obs(obs)
    print("Passenger is at: {0}, wants to go to {1}. Taxi currently at ({2}, {3})".format(
        obs_desc[obs_dict["passenger_location"]], 
        obs_desc[obs_dict["destination"]], 
        obs_dict["taxi_row"], 
        obs_dict["taxi_col"]))

'''@brief Takes an observation for the 'taxi-v3' environment and returns details observation space description
    @details returns a dict with "destination", "passenger_location", "taxi_col", "taxi_row"
    @see: https://gymnasium.farama.org/environments/toy_text/taxi/
'''
def breakdown_obs(obs):
    # ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination = X
    # X % 4 --> destination
    destination = obs % 4
    # X -= remainder, X /= 4
    obs -= destination
    obs /= 4
    # X % 5 --> passenger_location
    passenger_location = obs % 5
    # X -= remainder, X /= 5
    obs -= passenger_location
    obs /= 5
    # X % 5 --> taxi_col
    taxi_col = obs % 5
    # X -= remainder, X /=5 
    obs -= taxi_col
    # X --> taxi_row
    taxi_row = obs
    observation_dict= {
        "destination": destination, 
        "passenger_location": passenger_location,
        "taxi_row": taxi_row, 
        "taxi_col": taxi_col
    }
    return observation_dict


'''@brief simulate the environment with the agents taught policy
'''
def simulate_episodes(env, agent, num_episodes=3):
    for _ in range(num_episodes):
        done = False
        state, _ = env.reset()
        describe_obs(state)
        env.render()
        while not done:
            # Random choice from behavior policy
            action = agent.select_action(state)
            print("Action:", get_action_description(action))
            # take a step
            env.render()
            time.sleep(0.1)
            next_state, _, done, _, _ = env.step(action)
            state = next_state
        time.sleep(1.0)

def main():
    # Note: Use v3 for the latest version
    env = gym.make('Taxi-v3')
    num_obs, num_actions = describe_env(env)

    base_alpha = 0.1
    base_epsilon = 0.1
    gamma = 0.9
    episodes = 5000

    # TODO: Train
    print("\n=== Training with base hyperparameters ===")
    agent = Agent(num_obs, num_actions, alpha=base_alpha, epsilon=base_epsilon, gamma=gamma)
    returns, steps = agent.train(env, episodes)
    
    print("Total episodes:", episodes)
    print("Average return:", np.mean(returns))
    print("Average steps:", np.mean(steps))
    
    # Parameter variations
    alphas1 = [0.01, 0.001, 0.2]
    epsilons1 = [0.2, 0.3]

    results = {}

    print("\n=== Testing different learning rates ===")
    for a in alphas1:
        agent = Agent(num_obs, num_actions, alpha=a, epsilon=base_epsilon, gamma=gamma)
        returns, steps = agent.train(env, episodes)
        results[f"alpha={a}"] = (returns, steps)
        print(f"alpha={a}: avg return={np.mean(returns)}, avg steps={np.mean(steps)}")

    print("\n=== Testing different exploration factors ===")
    for e in epsilons1:
        agent = Agent(num_obs, num_actions, alpha=base_alpha, epsilon=e, gamma=gamma)
        returns, steps = agent.train(env, episodes)
        results[f"epsilon={e}"] = (returns, steps)
        print(f"epsilon={e}: avg return={np.mean(returns)}, avg steps={np.mean(steps)}")
        
    # Choose best parameters (you decide based on results)
    best_alpha = 0.2
    best_epsilon = 0.3
    
    print("\n=== Final training with chosen best parameters ===")
    agent = Agent(num_obs, num_actions, alpha=best_alpha, epsilon=best_epsilon, gamma=gamma)
    returns, steps = agent.train(env, episodes)
    
    print("Total episodes:", episodes)
    print("Average return:", np.mean(returns))
    print("Average steps:", np.mean(steps))
    
    # TODO: Simulate
    # Note how here, we change the render mode for testing/simulation
    env2 = gym.make('Taxi-v3', render_mode="human")
    simulate_episodes(env2, agent)

if __name__=="__main__":
    main()