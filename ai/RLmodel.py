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

class  PolicyNetwork(nn.Module):
    def __init__(self, state_dim,action_space):
        super(PolicyNetwork,self).__init__()
        self.action_space = action_space
        
        self.fc1 = nn.Linear(state_dim,128)
        self.fc2 = nn.Linear(128,128)

        

        self.output_layer = nn.Linear(128, sum(self.action_space.nvec))

        

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        logits = self.output_layer(x)

        #rozłożenie na action dim
        logits_split = torch.split(self.fc2(x), self.action_space, dim=1)
        # dla każdego wymiaru wylicz prawdopodobieństwa
        action_probs = [F.softmax(logit, dim=1) for logit in logits_split]
        return action_probs
        


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

