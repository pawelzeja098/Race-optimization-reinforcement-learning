import os
import sys

import sqlite3
import json
sys.path.append(r'E:\Programowanie\Reinforcement-learning-race-simulation')
import time
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

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_space):
        super(ActorCritic, self).__init__()
        
        # LayerNorm dla stabilności
        self.layer_norm = nn.LayerNorm(state_dim)
        
        # Shared layers
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        
        # BatchNorm
        self.ln1 = nn.LayerNorm(256) 
        self.ln2 = nn.LayerNorm(128)
        
        # Actor heads
        num_actions = action_space.nvec
        self.actor_heads = nn.ModuleList([
            nn.Linear(128, num_actions[i]) for i in range(len(num_actions))
        ])
        
        # Critic head
        self.critic = nn.Linear(128, 1)
        
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Xavier/Glorot initialization dla stabilności"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        #  Normalizacja inputu
        x = self.layer_norm(x)
        
        # Shared layers z ReLU
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        
        # Actor outputs (logits)
        action_logits = [head(x) for head in self.actor_heads]
        
        # Critic output
        value = self.critic(x)
        
        return action_logits, value



def select_action(model, state):
    """
    Select action based on current policy
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_list, state_value = model(state_tensor)

    actions = []
    log_probs = []

    for logits in logits_list:
        # ✅ WAŻNE: Używaj logits, NIE probs!
        dist = Categorical(logits=logits.squeeze(0))
        action = dist.sample()
        
        actions.append(action.item())
        log_probs.append(dist.log_prob(action))

    return actions, log_probs, state_value

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
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
        self.advantages = []  
        self.returns = []     

    def clear(self):
        self.__init__()



def ppo_update_batch(model, optimizer, buffer, batch_indices, device, clip_eps=0.2, entropy_coef=0.03, debug=False):
    """
    Algorytm PPO
    """
 
    if len(batch_indices) == 0:
        return 0.0, 0.0, 0.0
    
  
    states = torch.tensor(
        np.array([buffer.states[i] for i in batch_indices]), 
        dtype=torch.float32
    ).to(device)
    
    actions = torch.stack([buffer.actions[i] for i in batch_indices]).to(device)
    old_log_probs = torch.stack([buffer.log_probs[i] for i in batch_indices]).to(device)
    
   
    advantages = torch.tensor(
        [buffer.advantages[i] for i in batch_indices], 
        dtype=torch.float32
    ).to(device)
    
    returns = torch.tensor(
        [buffer.returns[i] for i in batch_indices], 
        dtype=torch.float32
    ).to(device)
    
  
    if len(advantages) > 1:  # Potrzebujemy przynajmniej 2 sampli dla std
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        
       
        if adv_std > 1e-6:
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        else:
            advantages = advantages - adv_mean  # Tylko centrowanie
    
    
    if len(returns) > 1:
        ret_mean = returns.mean()
        ret_std = returns.std()
        
        if ret_std > 1e-6:
            returns_normalized = (returns - ret_mean) / (ret_std + 1e-8)
        else:
            returns_normalized = returns - ret_mean
    else:
        returns_normalized = returns
    
   
    logits_list, values_new = model(states)
    
    values_new = values_new.squeeze(-1)  
    
 
    log_probs_new = []
    entropy_list = []
    
    for i, logits in enumerate(logits_list):
    
        dist = Categorical(logits=logits)
        
        acts = actions[:, i]
        
        log_probs_new.append(dist.log_prob(acts))
        entropy_list.append(dist.entropy())
    
    log_probs_new = torch.stack(log_probs_new).sum(dim=0)
    entropy = torch.stack(entropy_list).sum(dim=0).mean()
    
   
    ratio = torch.exp(log_probs_new - old_log_probs.detach())
    
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    

    actor_loss = -torch.min(surr1, surr2).mean()
    
    
    critic_loss = F.mse_loss(values_new, returns_normalized)
    

    loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy
    
   
    if torch.isnan(loss):
        print("⚠️ NaN detected in loss! Skipping update.")
        print(f"  Actor Loss: {actor_loss.item()}")
        print(f"  Critic Loss: {critic_loss.item()}")
        print(f"  Entropy: {entropy.item()}")
        print(f"  Advantages: min={advantages.min().item():.4f}, max={advantages.max().item():.4f}")
        print(f"  Returns: min={returns.min().item():.4f}, max={returns.max().item():.4f}")
        return 0.0, 0.0, 0.0
    

    optimizer.zero_grad()
    loss.backward()
    
  
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    
    optimizer.step()
    
    return actor_loss.item(), critic_loss.item(), entropy.item()


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
    
  
    x_no_scaled = raw_vector_x[no_scaler_x].reshape(1, -1)
    
    
    return np.hstack([x_no_scaled, x_min_max_scaled, x_robust_scaled]).flatten()

def save_checkpoint(model, optimizer, epoch, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
       
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint zapisany: {filename}")

def train_rl_model():
    """
    Trening modelu RL z wykorzystaniem PPO.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = RacingEnv()
    state_dim = env.observation_space.shape[0]
    model = ActorCritic(state_dim, env.action_space).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)  # Kompromis: 1e-4 za niskie, 3e-4 za wysokie
    buffer = RolloutBuffer()
    all_total_rewards = []
    all_epoch_rewards = []  # Nagroda na epokę (suma wyścigów)
    all_race_rewards = []   # Nagroda na pojedynczy wyścig
    num_epochs = 20 

    RACES_PER_EPOCH = 150  # 300 × 5 decyzji = ~1500 sampli (większy buffer)
    PPO_EPOCHS = 10
    BATCH_SIZE = 32  # 64/750 = 8.5% buffera (OK)
    
  
    ENTROPY_START = 0.01   
    ENTROPY_END = 0.001    


    try:
        scaler_minmax_X = joblib.load("models/scalerX_min_max_RL_final1.pkl")
        scaler_robust_X = joblib.load("models/scalerX_robust_RL_final1.pkl")
    except:
        data = load_data_from_db()
        X = create_x(data)
        scaler_minmax_X, scaler_robust_X = create_scalers(X)
        joblib.dump(scaler_minmax_X, "models/scalerX_min_max_RL_final1.pkl")
        joblib.dump(scaler_robust_X, "models/scalerX_robust_RL_final1.pkl")



    for epoch in range(num_epochs):
    
        progress = epoch / max(num_epochs - 1, 1)
        current_entropy_coef = ENTROPY_START + (ENTROPY_END - ENTROPY_START) * progress

        epoch_total_reward = 0
        
        
        race_configs = []
        
        if epoch < 10:
            # Faza 1: Długie wyścigi, usage 3.0, bez deszczu
            race_configs = [(2, 1, 1)] * RACES_PER_EPOCH
        else:
            # Faza 2: Długie wyścigi, usage 3.0, losowy deszcz (50/50)
            race_configs = [(2, 1, 0)] * (RACES_PER_EPOCH // 2) + [(2, 1, 1)] * (RACES_PER_EPOCH // 2)
        
        # Przetasuj konfiguracje
        random.shuffle(race_configs)
        
        for race_epoch, (i_time, i_usage, i_rain) in enumerate(race_configs):

            end_et = [734.0, 1032.0, 1932.0]
            total_steps = [1653, 2692, 5007]
            usage_multiplier = [1.0, 3.0]
            no_rain = [True, False]
            
           
            end_et = end_et[i_time]
            total_steps = total_steps[i_time]
            usage_mult = usage_multiplier[i_usage]
            no_rain = no_rain[i_rain]

            

            obs = env.reset(end_et = end_et,total_steps=total_steps,usage_multiplier=usage_mult,no_rain=no_rain)
            done = False  

            obs = scale_single_input(obs, scaler_minmax_X, scaler_robust_X)

            race_reward = 0
            
            while not done:
                # start = time.perf_counter()
                

                # Wybierz akcję na podstawie bieżącej obserwacji
                actions, log_prob, value = select_action(model, obs)
                
                # Wykonaj akcję. 
                next_obs, reward, done, _ = env.step(actions)

           
                # print(reward)
                reward /= 100.0  

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

            
            epoch_total_reward += race_reward
            all_race_rewards.append(race_reward)
            # Koniec epizodu (rolloutu)
        
       
        rewards_list = [r.item() for r in buffer.rewards]
        values_list = [v.item() for v in buffer.values]
        dones_list = buffer.dones
        
        advantages, returns = compute_gae(rewards_list, values_list, dones_list)
        
        # Zapisz w buforze (żeby ppo_update_batch miał dostęp)
        buffer.advantages = advantages
        buffer.returns = returns
        
        # PPO update z mini-batchami i wielokrotnymi epoch
        num_samples = len(buffer.states)
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        num_updates = 0
        
        for ppo_epoch in range(PPO_EPOCHS):
            # Losowo przemieszaj dane
            indices = np.random.permutation(num_samples)
            
            # Mini-batche
            for batch_idx, start in enumerate(range(0, num_samples, BATCH_SIZE)):
                end = min(start + BATCH_SIZE, num_samples)
                batch_indices = indices[start:end]
                
                # DEBUG: Pierwszy batch PPO epoch 5 (środek treningu)
                debug_mode = (ppo_epoch == 5 and batch_idx == 0)
                
                actor_loss, critic_loss, entropy = ppo_update_batch(
                    model, optimizer, buffer, batch_indices, device, 
                    entropy_coef=current_entropy_coef, debug=debug_mode
                )
                
                total_actor_loss += actor_loss
                total_critic_loss += critic_loss
                total_entropy += entropy
                num_updates += 1
        
        avg_actor_loss = total_actor_loss / num_updates
        avg_critic_loss = total_critic_loss / num_updates
        avg_entropy = total_entropy / num_updates
        
        all_epoch_rewards.append(epoch_total_reward)
        
        # Logowanie
        if epoch % 1 == 0:
            avg_race_reward = epoch_total_reward / RACES_PER_EPOCH
            recent_avg = np.mean(all_race_rewards[-100:]) if len(all_race_rewards) >= 100 else np.mean(all_race_rewards)
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"  Epoch Total: {epoch_total_reward:.4f} (z {RACES_PER_EPOCH} wyścigów)")
            print(f"  Avg per race: {avg_race_reward:.4f}")
            print(f"  Last 100 races avg: {recent_avg:.4f}")
            print(f"  Buffer size: ~{RACES_PER_EPOCH * 5} decyzji")  
            print(f"  Actor Loss: {avg_actor_loss:.4f}")
            print(f"  Critic Loss: {avg_critic_loss:.4f}")
            print(f"  Entropy: {avg_entropy:.4f}")
            print(f"  Entropy Coef: {current_entropy_coef:.4f}")
            print(f"{'='*60}")

            #zapsi logi do pliku
            with open("ai/rl_learning_plot/rl_training_log.txt", "a", encoding='utf-8') as log_file:
                log_file.write(f"Epoch {epoch}/{num_epochs}\n")
                log_file.write(f"  Epoch Total: {epoch_total_reward:.4f} (z {RACES_PER_EPOCH} wyscigow)\n")
                log_file.write(f"  Avg per  race: {avg_race_reward:.4f}\n")
                log_file.write(f"  Last 100 races avg: {recent_avg:.4f}\n")
                log_file.write(f"  Buffer size: ~{RACES_PER_EPOCH * 5} decyzji\n")
                log_file.write(f"  Actor Loss: {avg_actor_loss:.4f}\n")
                log_file.write(f"  Critic Loss: {avg_critic_loss:.4f}\n")
                log_file.write(f"  Entropy: {avg_entropy:.4f}\n")
                log_file.write(f"  Entropy Coef: {current_entropy_coef:.4f} (zmniejsza sie)\n")
                log_file.write(f"{'='*60}\n")
        
        # Wyczyść bufor
        buffer.clear()
        
        # Checkpoint co 5 epok
        if epoch % 5 == 0 and epoch > 0:
            save_checkpoint(model, optimizer, epoch, f"models/RL_agent_final_epoch1{epoch}.pth")
    
    # Finalny checkpoint
    save_checkpoint(model, optimizer, num_epochs, "models/RL_agent_final1.pth")
    
    # ✅ Wykres POJEDYNCZYCH WYŚCIGÓW (prawdziwy postęp)
    plt.figure(figsize=(14, 6))
    plt.plot(all_race_rewards, alpha=0.3, label='Pojedynczy wyścig')
    
    # Średnia krocząca z 50 wyścigów
    rolling_avg = pd.Series(all_race_rewards).rolling(window=50).mean()
    plt.plot(rolling_avg, color='red', linewidth=2, label='Średnia krocząca (50)')
    
    plt.title('Postęp Uczenia (Nagroda za Pojedynczy Wyścig)')
    plt.xlabel('Numer wyścigu')
    plt.ylabel('Nagroda (znormalizowana)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('ai/rl_learning_plot/rl_reward_per_race_final.png', dpi=150)
    plt.show()

train_rl_model()

# import torch
# from torchview import draw_graph
# import graphviz
# from gymnasium import spaces 

# # 1. Konfiguracja modelu

# action_space = spaces.MultiDiscrete([
#                                 2, # Pit stop or not
#                                 # 2, # Confirm pit stop or not
#                                 5, # Tire change (0-4) No,#  soft, medium, hard, wet
#                                 2, # Repair or not (0-1)
#                                 6, # Fuel#  * 0.2 (0-20)
#                                     ])
# model = ActorCritic(state_dim=26, action_space=action_space)

# # 2. Przygotowanie da# nych (X oraz # Stan)
# dummy_x = torch.randn(1, 1, 26)
# # Stan dla#  LSTM: (h_0, c_# 0) -> rozmiar [layers, ba# tch, hidden]


# # 3. Generowanie grafu
# model_graph = draw_graph(
#     model, 
#     input_data=dummy_x,  # Przekazujemy X i Stan
#     depth=1, 
#     expand_nested=True,
#     save_graph=False  # Nie zapisuj jeszcze automatycznie
# )


# print(model_graph.visual_graph.source)

# from graphviz import Digraph
# from graphviz import Digraph

# dot = Digraph(comment='PPO - Główne Fazy', format='png')
# dot.attr(rankdir='LR', size='12,6')  # Poziomo!
# dot.attr('node', shape='box', style='rounded,filled', fontname='Arial', fontsize='12')

# # 3 główne fazy
# with dot.subgraph(name='cluster_rollout') as c:
#     c.attr(style='filled', color='lightgrey', label='FAZA 1: Rollout')
#     c.node('r1', '50 wyścigów\n× ~1000 kroków')
#     c.node('r2', 'Zbieranie:\n(s, a, r, V(s))')
#     c.edge('r1', 'r2')

# with dot.subgraph(name='cluster_gae') as c:
#     c.attr(style='filled', color='lightgreen', label='FAZA 2: GAE')
#     c.node('g1', 'Obliczenie\nδ_t dla każdego kroku')
#     c.node('g2', 'Akumulacja wstecz:\nÂ_t = Σ(γλ)^l δ')
#     c.edge('g1', 'g2')

# with dot.subgraph(name='cluster_ppo') as c:
#     c.attr(style='filled', color='lightblue', label='FAZA 3: PPO Update')
#     c.node('p1', '10 epok\n× mini-batche (64)')
#     c.node('p2', 'L = -L^CLIP\n+ 0.5·L^VF\n- 0.05·H')
#     c.node('p3', 'Backprop\n+ grad clip')
#     c.edge('p1', 'p2')
#     c.edge('p2', 'p3')

# # Połączenia między fazami
# dot.edge('r2', 'g1', label='Buffer\n~50k kroków')
# dot.edge('g2', 'p1', label='Advantages\n+ Returns')
# dot.edge('p3', 'r1', label='Nowa epoka', style='dashed')

# print(dot.source)
# print("Zapisano: ppo_simplified.png")
