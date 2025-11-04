import os
import sys
sys.path.append(r'E:\Programowanie\Reinforcement-learning-race-simulation')

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # [1, obs_dim]
    action_probs_list, state_value = model(state)  # lista tensorów [1, num_options]

    actions = []
    log_probs = []

    for probs in action_probs_list:
        probs = probs.squeeze(0)  # usuń batch dimension -> [num_options]
        action = torch.multinomial(probs, num_samples=1).item()
        actions.append(action)
        log_probs.append(torch.log(probs[action]))

    return actions, log_probs, state_value

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

class RolloutBuffer:
    """
    Buffer to store trajectories for PPO updates. One race worth of data.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def clear(self):
        self.__init__()

def ppo_update(model, optimizer, buffer, clip_eps=0.2):
    """
    Proximal Policy Optimization (PPO) update step.
    Clipping the policy update to avoid large updates.

    """
    states = torch.tensor(buffer.states, dtype=torch.float32)
    old_log_probs = torch.stack(buffer.log_probs)
    actions = buffer.actions
    rewards = buffer.rewards
    values = torch.stack(buffer.values).squeeze()
    dones = buffer.dones

    advantages, returns = compute_gae(rewards, values.tolist(), dones)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32)

    # Forward pass
    action_probs, values_new = model(states)

    # Oblicz log_probs dla MultiDiscrete
    log_probs_new = []
    for i, probs in enumerate(action_probs):
        dist = torch.distributions.Categorical(probs)
        acts = torch.tensor([a[i] for a in actions])
        log_probs_new.append(dist.log_prob(acts))
    log_probs_new = torch.stack(log_probs_new).sum(dim=0)

    # PPO ratio
    ratio = torch.exp(log_probs_new - old_log_probs.detach())
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantages

    actor_loss = -torch.min(surr1, surr2).mean()  # polityka
    critic_loss = F.mse_loss(values_new.squeeze(), returns)  # Critic
    loss = actor_loss + 0.5 * critic_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

env = RacingEnv()
state_dim = env.observation_space.shape[0]
model = ActorCritic(state_dim, env.action_space)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
buffer = RolloutBuffer()

num_epochs = 1000
# steps_per_epoch = 200 # Ta zmienna nie jest używana w tej logice

for epoch in range(num_epochs):
    
    obs = env.reset()
    done = False  # <-- POPRAWKA 1: Zainicjuj 'done'
    
    while not done:
        
        # Wybierz akcję na podstawie bieżącej obserwacji
        actions, log_prob, value = select_action(model, obs)
        
        # Wykonaj akcję. 
        # Zmieniono env.step(actions, obs) na env.step(actions)
        # Środowisko samo śledzi swój stan (obs), nie trzeba go podawać.
        next_obs, reward, done, _ = env.step(actions, obs)

        # --- POPRAWKA 2: Zapisuj 'obs' (stan s_t) ---
        buffer.states.append(obs) 
        # ---------------------------------------------

        buffer.actions.append(actions)
        buffer.log_probs.append(log_prob)
        buffer.values.append(value)
        buffer.rewards.append(torch.tensor(reward, dtype=torch.float32))
        buffer.dones.append(done)

        # Zaktualizuj stan na następną iterację
        obs = next_obs
        
        # --- POPRAWKA 3: Usunięto blok 'if done: reset()' ---
        # Pętla 'while' sama się zakończy, a reset nastąpi
        # na początku następnej epoki.

    # Koniec epizodu (rolloutu)
    
    # PPO update po zebraniu danych z całego epizodu
    ppo_update(model, optimizer, buffer)
    
    if epoch % 10 == 0:
        total_reward = sum([r.item() for r in buffer.rewards])
        print(f"Epoch {epoch}, total reward {total_reward}")
    
    # Wyczyść bufor po aktualizacji, gotowy na nową epokę
    buffer.clear()



# env = RacingEnv()

# obs_dim = env.observation_space.shape[0]  # lub inny rozmiar zgodny z obserwacją
# action_space = env.action_space  # Twoje rozmiary akcji

# model = PolicyNetwork(obs_dim,action_space)
# obs = env.reset()
# done = False
# env.step(action_space)
# obs = env.state
# while not done:
#     obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # dodaj batch

#     # Weź rozkłady prawdopodobieństw
#     action_probs = model(obs_tensor)

#     # Wylosuj akcje z rozkładów
#     actions = [torch.multinomial(probs, num_samples=1).item() for probs in action_probs]

#     # Zrealizuj akcję w środowisku
#     obs, reward, done, info = env.step(actions)

