import sys
sys.path.append(r'E:\Programowanie\Reinforcement-learning-race-simulation')

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
from envs.racing_env import RacingEnv

class  ActorCritic(nn.Module):
    def __init__(self, state_dim,action_space):
        super(ActorCritic,self).__init__()
        self.action_space = action_space

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)

        

        
        #Actor Every action dim has its own output layer
        self.output_layers = nn.ModuleList()
        for n in self.action_space.nvec:
            self.output_layers.append(nn.Linear(128, n))

        #Critic
        self.critic = nn.Linear(128, 1)

        

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        logits = [layer(x) for layer in self.output_layers] # list of logits for each action dim
        
        #rozłożenie na action dim
        # logits_split = torch.split(self.fc2(x), self.action_space, dim=1)
        # dla każdego wymiaru wylicz prawdopodobieństwa
        action_probs = [F.softmax(logit, dim=1) for logit in logits]
        state_value = self.critic(x)
        return action_probs, state_value

def select_action(model,state):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0) #add state batch dimension
    action_probs, state_value = model(state) # get action probabilities and state value

    actions = []
    log_probs = []

    for probs in action_probs:
        action = torch.multinomial(probs, num_samples=1).item()
        actions.append(action)
        log_probs.append(torch.log(probs[action]))

    return actions, action_probs, state_value

def compute_gae(rewards, values, dones, gamma=0.995, lam=0.95):
    """
    Compute Generalized Advantage Estimation (GAE)
    See how good was action compared to average (value)
    Short term and long term rewards(TD + Monte Carlo)
    Calculating after race is done - backward through time
    Args:
        rewards: list of rewards
        values: list of value estimates
        dones: list of done flags
        gamma: discount factor
        lam: GAE parameter
    Returns:
        advantages: list of advantage estimates
        returns: list of return estimates (advantages + values)

    """
    advantages = []
    gae = 0
    values = values + [0]  # bootstrap dla końca epizodu
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns

env = RacingEnv()

obs_dim = env.observation_space.shape[0]  # lub inny rozmiar zgodny z obserwacją
action_space = env.action_space  # Twoje rozmiary akcji

model = PolicyNetwork(obs_dim,action_space)
obs = env.reset()
done = False
env.step(action_space)
obs = env.state
while not done:
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # dodaj batch

    # Weź rozkłady prawdopodobieństw
    action_probs = model(obs_tensor)

    # Wylosuj akcje z rozkładów
    actions = [torch.multinomial(probs, num_samples=1).item() for probs in action_probs]

    # Zrealizuj akcję w środowisku
    obs, reward, done, info = env.step(actions)

