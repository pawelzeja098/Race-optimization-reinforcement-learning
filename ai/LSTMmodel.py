import joblib
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.model_selection import LeaveOneOut
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
# from envs.filtr_json_from_race import load_from_db
import sqlite3
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score

class LSTMStatePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,n_steps_ahead, num_layers=1):
        super(LSTMStatePredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size * n_steps_ahead)
        self.scaler_X = None
        self.scaler_Y = None
        self.n_steps_ahead = n_steps_ahead
        self.output_size = output_size

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), self.lstm.hidden_size).to(x.device) #hidden state
        c_0 = torch.zeros(1, x.size(0), self.lstm.hidden_size).to(x.device) #cell state

        out, _ = self.lstm(x, (h_0, c_0))
        # out = self.fc(out[:, -1, :])  # Use the last time step's output
        out = self.fc(out[:, -1, :]).view(-1, self.n_steps_ahead, self.output_size)
        return out

def create_scalers(X,Y):

    cont_indices_x = slice(0, 18)   # continuous columns for X (0–17)
    cont_indices_y = slice(0, 12)   # continuous columns for Y (0–11)

    # Scale continuous features
    flat_x = np.vstack([x[:, cont_indices_x] for x in X])
    flat_y = np.vstack([y[:, cont_indices_y] for y in Y])

    scaler_X = MinMaxScaler().fit(flat_x)
    scaler_Y = MinMaxScaler().fit(flat_y)
    return scaler_X, scaler_Y


def scale_input(X, Y, scaler_X, scaler_Y):
    cont_indices_x = slice(0, 18)   # continuous columns for X
    cont_indices_y = slice(0, 12)   # continuous columns for Y

    X_scaled_grouped = []
    Y_scaled_grouped = []

    for x_seq, y_seq in zip(X, Y):
        x_scaled = np.array(x_seq, dtype=float)
        x_scaled[:, cont_indices_x] = scaler_X.transform(x_seq[:, cont_indices_x])
        X_scaled_grouped.append(x_scaled)

        y_scaled = np.array(y_seq, dtype=float)
        y_scaled[:, cont_indices_y] = scaler_Y.transform(y_seq[:, cont_indices_y])
        Y_scaled_grouped.append(y_scaled)

    # Conversion to torch tensors
    # X_t = [torch.tensor(x, dtype=torch.float32) for x in X_scaled_grouped]
    # Y_cont_t = [torch.tensor(y[:, cont_indices_y], dtype=torch.float32) for y in Y_scaled_grouped]

    return X_scaled_grouped, Y_scaled_grouped