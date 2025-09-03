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

# policy = PolicyNetwork(obs_dim, action_space)


# obs = torch.tensor([observation], dtype=torch.float32)

# # wyliczamy rozkłady dla każdego wymiaru
# action_probs = model(obs)

# # losujemy akcję dla każdego wymiaru
# actions = [torch.multinomial(probs, num_samples=1).item() for probs in action_probs]

# class DQN(nn.Module):
#     def __init__(self, state_dim,action_dim):
#         #call nn.Module consturctor
#         super(DQN,self).__init__()

#         #network layers
#         self.fc = nn.Sequential(
#             #linear layer
#             nn.Linear(state_dim,128),
#             #activation function to add non-linearity
#             nn.ReLU(),
#             nn.Linear(128,128),
#             #activation function to add non-linearity
#             nn.ReLU(),
#             nn.Linear(128,action_dim)
#         )
#     #pass through network
#     def forward(self,x):
#         return self.fc(x)

# #Store examples from past
# class ReplayBuffer:
#     #max storage = capacity
#     def __init__(self, capacity=10000):
#         self.buffer = deque(maxlen=capacity)
#     #add experience to buffor
#     def push(self, state, action, reward, next_state, done):
#         self.buffer.append((state, action, reward, next_state, done))
#     #get random sample from buffor
#     def sample(self, batch_size=64):
#         batch = random.sample(self.buffer, batch_size)
#         states, actions, rewards, next_states, dones = zip(*batch)
#         return (
#             torch.tensor(states, dtype=torch.float32),
#             torch.tensor(actions),
#             torch.tensor(rewards, dtype=torch.float32),
#             torch.tensor(next_states, dtype=torch.float32),
#             torch.tensor(dones, dtype=torch.float32)
#         )
# ##DALEJ PRZECZYTAC I OPISAC
#     def __len__(self):
#         return len(self.buffer)
    





#     env = ... # Twoje środowisko np. symulacja pitstopów
# model = DQN(state_dim=env.state_size, action_dim=env.action_space)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# replay_buffer = ReplayBuffer()
# gamma = 0.99  # współczynnik przyszłej nagrody

# for episode in range(1000):
#     state = env.reset()
#     done = False
#     while not done:
#         # Epsilon-greedy
#         if random.random() < epsilon:
#             action = random.randint(0, env.action_space - 1)
#         else:
#             with torch.no_grad():
#                 q_values = model(torch.tensor(state).float())
#                 action = q_values.argmax().item()

#         next_state, reward, done, _ = env.step(action)
#         replay_buffer.push(state, action, reward, next_state, done)
#         state = next_state

#         # Trening modelu
#         if len(replay_buffer) > 64:
#             states, actions, rewards, next_states, dones = replay_buffer.sample()

#             # Q(s,a)
#             q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

#             # max_a' Q(s',a')
#             next_q_values = model(next_states).max(1)[0]
#             target_q_values = rewards + gamma * next_q_values * (1 - dones)

#             loss = nn.MSELoss()(q_values, target_q_values.detach())
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()



# class PPOActorCritic(nn.Module):
#     def __init__(self, state_dim, action_dims):
#         super(PPOActorCritic, self).__init__()
#         self.shared = nn.Sequential(
#             nn.Linear(state_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU()
#         )
        
#         # Dla każdej akcji (np. opony, moc) mamy osobny softmax
#         self.actor_heads = nn.ModuleList([
#             nn.Linear(128, dim) for dim in action_dims
#         ])
        
#         self.critic = nn.Linear(128, 1)

#     def forward(self, x):
#         shared = self.shared(x)
#         # Zwraca listę dystrybucji softmax dla każdej decyzji
#         action_probs = [F.softmax(head(shared), dim=-1) for head in self.actor_heads]
#         state_value = self.critic(shared)
#         return action_probs, state_value