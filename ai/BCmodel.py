"""
Pretraining model using Behavioral Cloning (BC) approach.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from envs.racing_env_old import RacingEnv
import numpy as np
import json
# from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import random

class BCModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(BCModel, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        

        # Każda dyskretna akcja ma własną warstwę wyjściową
        self.output_layers = nn.ModuleList()
        for n in action_dim:
            self.output_layers.append(nn.Linear(128, n))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output_layer(x)
    
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    states = []
    actions = []
    
    for record in data:
        state = record['state']
        action = record['action']
        
        states.append(state)
        actions.append(action)
    
    return np.array(states), np.array(actions)

def save_model(model, path):
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    env = RacingEnv()
    
    model = BCModel(state_dim=22, action_dim=[2,5,2,23])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    batch_size = 64
    num_epochs = 20
