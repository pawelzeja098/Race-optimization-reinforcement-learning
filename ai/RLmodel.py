import os
import sys

import sqlite3
import json
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
from sklearn.preprocessing import StandardScaler , MinMaxScaler, RobustScaler
import joblib
from config import Y_INDEXES

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

def load_data_from_db():
    
    """
    Load data so that each race is a separate sequence:
    X = [ [state1_race1, state2_race1, ...], [state1_race2, ...] ]
    Y = [ [next1_race1, next2_race1, ...], ... ]
    """
    conn = sqlite3.connect(
        "E:/pracadyp/Race-optimization-reinforcement-learning/data/db_states_for_regress/race_data_states.db"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT race_id, states_json FROM races ORDER BY race_id")
    rows = cursor.fetchall()
    conn.close()


    data = []

    for race_id, states_json in rows:
        states = json.loads(states_json)
        data.append(states)
    
    return data


def create_x(data):
    X_grouped = []

    for race in data:
        X_seq = []
        for i in range(len(race) - 1):
            X_seq.append(race[i][:Y_INDEXES])  # current state
        # dodajemy każdy wyścig osobno
        X_grouped.append(np.array(X_seq, dtype=float))
    
    rl_feature_indices = np.r_[2:9, 11:12, 16:25, 26:33, 34:37]
    X_filtered = []
    for race in X_grouped:
        # race ma kształt (N_steps, 38)
        # Bierzemy wszystkie wiersze (:), ale tylko wybrane kolumny (indices)
        race_subset = race[:, rl_feature_indices]
        X_filtered.append(race_subset)
        
    del X_grouped
        
    return X_filtered



def create_scalers(X):

    # cont_indices_x = slice(0, CONT_LENGTH)   # continuous columns for X (0–18)
    # cont_indices_y = slice(0, Y_SHAPE)   # continuous columns for Y (0–11)

    no_scaler_x = slice(0, 8)  # no scaler for X
    min_max_scaler_x = slice(8, 20)  # min-max scaler for X
    robust_scaler_x = slice(20, 28)  # robust scaler for X
     # robust scaler for Y

   
    flat_x_min_max = np.vstack([x[:, min_max_scaler_x] for x in X])
    flat_x_robust = np.vstack([x[:, robust_scaler_x] for x in X])


    scaler_X_min_max = MinMaxScaler().fit(flat_x_min_max)
    scaler_X_robust = RobustScaler().fit(flat_x_robust)

    return scaler_X_min_max, scaler_X_robust

def scale_single_input(raw_vector_x, scaler_X_min_max, scaler_X_robust):
    """
    Skaluje pojedynczy wektor (37,), stosując scaler tylko do 
    części ciągłej (0-19) i zostawiając kategorialną (20-36).
    """
    no_scaler_x = slice(0, 8)  # no scaler for X
    min_max_scaler_x = slice(8, 20)  # min-max scaler for X
    robust_scaler_x = slice(20, 28)  # robust scaler for X
 
    
    # raw_vector_x[cont_indices_x] ma kształt (19,)
    # Musimy go przekształcić na (1, 19) dla scalera
    x_min_max_scaled = scaler_X_min_max.transform([raw_vector_x[min_max_scaler_x]])
    x_robust_scaled = scaler_X_robust.transform([raw_vector_x[robust_scaler_x]])
    
    # raw_vector_x[cat_indices_x] ma kształt (18,)
    # --- POPRAWKA TUTAJ ---
    # Musimy go przekształcić na (1, 19), aby pasował do hstack
    x_no_scaled = raw_vector_x[no_scaler_x].reshape(1, -1)
    
    # Teraz łączymy (1, 19) z (1, 18) -> (1, 37)
    # i spłaszczamy z powrotem do 1D (37,)
    return np.hstack([x_no_scaled, x_min_max_scaled, x_robust_scaled]).flatten()

def save_checkpoint(model, optimizer, epoch, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # Możesz dodać inne rzeczy, np. historię nagród
        # 'all_total_rewards': all_total_rewards 
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint zapisany: {filename}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = RacingEnv()
state_dim = env.observation_space.shape[0]
model = ActorCritic(state_dim, env.action_space).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
buffer = RolloutBuffer()
all_total_rewards = []
num_epochs = 500

RACES_PER_EPOCH = 10


try:
    scaler_minmax_X = joblib.load("models/scalerX_min_max_RL.pkl")
    scaler_robust_X = joblib.load("models/scalerX_robust_RL.pkl")
except:
    data = load_data_from_db()
    X = create_x(data)
    scaler_minmax_X, scaler_robust_X = create_scalers(X)
    joblib.dump(scaler_minmax_X, "models/scalerX_min_max_RL.pkl")
    joblib.dump(scaler_robust_X, "models/scalerX_robust_RL.pkl")



for epoch in range(num_epochs):

    epoch_total_reward = 0
    
    for race_epoch in range(RACES_PER_EPOCH):

        end_et = [734.0,1032.0,1932.0] # 11592.0
        total_steps = [1653,2692,5007] # 30922
        usage_multiplier = [1.0,3.0]
        no_rain = [True,False]

        i_time = random.choice([0,1,2])
        i_usage = random.choice([0,1])
        i_rain = random.choice([0,1])
        end_et = end_et[i_time]
        total_steps = total_steps[i_time]
        usage_mult = usage_multiplier[i_usage]
        no_rain = no_rain[i_rain]

        obs = env.reset(end_et = end_et,total_steps=total_steps,usage_multiplier=usage_mult,no_rain=no_rain)
        done = False  

        obs = scale_single_input(obs, scaler_minmax_X, scaler_robust_X)

        race_reward = 0
        
        while not done:
            

            # Wybierz akcję na podstawie bieżącej obserwacji
            actions, log_prob, value = select_action(model, obs)
            
            # Wykonaj akcję. 
            next_obs, reward, done, _ = env.step(actions)

            next_obs = scale_single_input(next_obs, scaler_minmax_X, scaler_robust_X)

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
            race_reward += reward
            
            # --- POPRAWKA 3: Usunięto blok 'if done: reset()' ---
            # Pętla 'while' sama się zakończy, a reset nastąpi
            # na początku następnej epoki.
        epoch_total_reward += race_reward
        # Koniec epizodu (rolloutu)
        
        # PPO update po zebraniu danych z całego epizodu
        ppo_update(model, optimizer, buffer, device)
        total_reward = sum([r.item() for r in buffer.rewards])
        all_total_rewards.append(total_reward)
        
        # if epoch % 10 == 0:
            
        print(f"Epoch {epoch}, total reward {total_reward}")
        print(buffer.actions)
        print(buffer.rewards)
        
        # Wyczyść bufor po aktualizacji, gotowy na nową epokę
        buffer.clear()

save_checkpoint(model, optimizer, epoch, "models/RL_agent.pth")


print("Trening zakończony. Rysowanie wykresu nagrody...")

plt.figure(figsize=(12, 6))
plt.plot(all_total_rewards, label='Całkowita nagroda')

# --- (Opcjonalnie, ale BARDZO ZALECANE) Wygładzony wykres ---
# Nagrody w RL bardzo "skaczą". Średnia krocząca pokazuje prawdziwy trend.
rolling_avg = pd.Series(all_total_rewards).rolling(window=50).mean() # Średnia z 50 epok
plt.plot(rolling_avg, label='Średnia krocząca', color='red', linewidth=2)
# -----------------------------------------------------------

plt.title('Postęp Uczenia Modelu RL (Nagroda na Epokę)')
plt.xlabel('Epoka')
plt.ylabel('Całkowita Nagroda')
plt.legend()
plt.grid(True)
plt.savefig(f'ai/rl_training_race_historyplots/rl_reward.png', dpi=150)


