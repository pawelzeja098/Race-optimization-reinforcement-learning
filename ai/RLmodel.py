import os
import sys
sys.path.append(r'E:\Programowanie\Reinforcement-learning-race-simulation')

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import pandas as pd
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

# def select_action(model,state):
#     state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)  # [1, obs_dim]
#     action_probs_list, state_value = model(state)  # lista tensorów [1, num_options]

#     actions = []
#     log_probs = []

#     for probs in action_probs_list:
#         probs = probs.squeeze(0)  # usuń batch dimension -> [num_options]
#         action = torch.multinomial(probs, num_samples=1).item()
#         actions.append(action)
#         log_probs.append(torch.log(probs[action]))

#     return actions, log_probs, state_value

def select_action(model, state):
    """
    Poprawnie wybiera akcję, używając obiektu Categorical,
    aby zapewnić spójność z pętlą 'ppo_update'.
    """
    
    # Przenieś stan na 'device'
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    # Użyj torch.no_grad() do inferencji (szybciej, oszczędza pamięć)
    with torch.no_grad():
        # Model zwraca LISTĘ LOGITÓW i wartość stanu
        logits_list, state_value = model(state_tensor)

    actions = []
    log_probs = []

    for logits in logits_list:
        # 1. Stwórz obiekt dystrybucji z logitów
        #    (squeeze(0) usuwa wymiar batcha)
        dist = Categorical(logits=logits.squeeze(0))
        
        # 2. WYLOSUJ akcję (to jest klucz do EKSPLORACJI)
        action = dist.sample() # Zwraca tensor, np. tensor(1)
        
        # 3. Zapisz akcję jako zwykłą liczbę (dla env.step)
        actions.append(action.item()) # Np. 1
        
        # 4. Zapisz log_prob dla tej wylosowanej akcji
        log_probs.append(dist.log_prob(action)) # Zwraca tensor, np. tensor(-0.45)

    # Zwraca: listę liczb, listę tensorów, tensor
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

def ppo_update(model, optimizer, buffer, device, clip_eps=0.2):
    """
    Poprawiona funkcja PPO update.
    """
    
    # --- 1. Konwersja danych z bufora na tensory GPU ---
    
    states = torch.tensor(np.array(buffer.states), dtype=torch.float32).to(device)
    
    # POPRAWKA BŁĘDU 2: Przekonwertuj listę tensorów akcji [tensor([0,1]), ...]
    # na jeden duży tensor (N_kroków, N_akcji) na GPU
    actions = torch.stack(buffer.actions).to(device) 
    
    old_log_probs = torch.stack(buffer.log_probs).to(device)
    # values = torch.stack(buffer.values).squeeze().to(device)
    values = torch.stack(buffer.values).reshape(-1).to(device)
    
    
    # POPRAWKA BŁĘDU 3: Przekonwertuj listę tensorów nagród na listę float
    rewards_list = [r.item() for r in buffer.rewards]
    dones_list = buffer.dones

    # --- 2. Obliczenia na CPU (GAE) ---
    advantages, returns = compute_gae(rewards_list, values.tolist(), dones_list)
    advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)

    # --- 3. Forward pass i obliczenie strat ---
    
    # action_probs to lista [tensor_dist_pit, tensor_dist_opony]
    action_probs, values_new = model(states) 

    # Oblicz log_probs dla MultiDiscrete
    log_probs_new = []
    for i, probs in enumerate(action_probs):
        dist = torch.distributions.Categorical(probs)
        
        # POPRAWKA BŁĘDU 1: Użyj "krojonego" tensora 'actions' z GPU
        # 'actions' ma kształt (N_kroków, N_akcji)
        # 'acts' będzie miało kształt (N_kroków,)
        acts = actions[:, i] 
        
        # dist (GPU) i acts (GPU) są teraz na tym samym urządzeniu
        log_probs_new.append(dist.log_prob(acts))
        
    log_probs_new = torch.stack(log_probs_new).sum(dim=0)

    # PPO ratio
    ratio = torch.exp(log_probs_new - old_log_probs.detach())
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantages

    actor_loss = -torch.min(surr1, surr2).mean()
    critic_loss = F.mse_loss(values_new.squeeze(), returns)
    loss = actor_loss + 0.5 * critic_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Zwróć straty do logowania
    return actor_loss.item(), critic_loss.item()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = RacingEnv()
state_dim = env.observation_space.shape[0]
model = ActorCritic(state_dim, env.action_space).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
buffer = RolloutBuffer()
all_total_rewards = []
num_epochs = 250
last_step = 0
# steps_per_epoch = 200 # Ta zmienna nie jest używana w tej logice

for epoch in range(num_epochs):
    
    obs = env.reset()
    done = False  # <-- POPRAWKA 1: Zainicjuj 'done'
    
    while not done:
        
        # Wybierz akcję na podstawie bieżącej obserwacji
        actions, log_prob, value = select_action(model, obs)
        
        # Wykonaj akcję. 
        next_obs, reward, done, last_step, _ = env.step(actions, last_step)

        # --- POPRAWKA 2: Zapisuj 'obs' (stan s_t) ---
        buffer.states.append(obs) 
        # ---------------------------------------------

        if isinstance(actions, list):
            # Połącz listę tensorów akcji w jeden tensor [0, 1]
            actions_tensor = torch.tensor(actions) 
            # Zsumuj log-prawdopodobieństwa dla wspólnej akcji
            log_prob_tensor = sum(log_prob)
        else:
            # Jeśli to nie lista, po prostu użyj wartości
            actions_tensor = actions
            log_prob_tensor = log_prob

        buffer.actions.append(actions_tensor)
        buffer.log_probs.append(log_prob_tensor)
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
    ppo_update(model, optimizer, buffer, device)
    total_reward = sum([r.item() for r in buffer.rewards])
    all_total_rewards.append(total_reward)
    
    # if epoch % 10 == 0:
        
    print(f"Epoch {epoch}, total reward {total_reward}")
    print(buffer.actions)
    
    # Wyczyść bufor po aktualizacji, gotowy na nową epokę
    buffer.clear()


print("Trening zakończony. Rysowanie wykresu nagrody...")

plt.figure(figsize=(12, 6))
plt.plot(all_total_rewards, label='Całkowita nagroda (Surowa)')

# --- (Opcjonalnie, ale BARDZO ZALECANE) Wygładzony wykres ---
# Nagrody w RL bardzo "skaczą". Średnia krocząca pokazuje prawdziwy trend.
rolling_avg = pd.Series(all_total_rewards).rolling(window=50).mean() # Średnia z 50 epok
plt.plot(rolling_avg, label='Średnia krocząca (wygładzona)', color='red', linewidth=2)
# -----------------------------------------------------------

plt.title('Postęp Uczenia Modelu RL (Nagroda na Epokę)')
plt.xlabel('Epoka')
plt.ylabel('Całkowita Nagroda')
plt.legend()
plt.grid(True)
plt.show()

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

