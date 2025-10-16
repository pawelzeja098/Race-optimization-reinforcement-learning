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
from torch.optim.lr_scheduler import ReduceLROnPlateau

class LSTMStatePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,n_steps_ahead, num_layers=1):
        super(LSTMStatePredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # self.fc = nn.Linear(hidden_size, output_size * n_steps_ahead)

        self.heads = nn.ModuleList([
        nn.Linear(hidden_size, 2),  # progress
        nn.Linear(hidden_size, 1),  # fuel
        nn.Linear(hidden_size, 4),  # wear
        nn.Linear(hidden_size, 4),  # temp
        nn.Linear(hidden_size, 1)   # track wetness
        ])

        self.scaler_X = None
        self.scaler_Y = None
        self.n_steps_ahead = n_steps_ahead
        self.output_size = output_size

    def forward(self, x):
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h_0, c_0))

        # Bierzemy ukryty stan z ostatniego kroku czasowego
        last_hidden = out[:, -1, :]  # shape [batch, hidden_size]

        # Każdy head przewiduje własną grupę cech
        outputs = [head(last_hidden) for head in self.heads]  

        # Łączymy wszystko w jeden wektor
        combined = torch.cat(outputs, dim=1)  # [B, 12]

      
        combined = combined.unsqueeze(1).repeat(1, self.n_steps_ahead, 1)
        

        return combined

def create_scalers(X,Y):

    cont_indices_x = slice(0, 19)   # continuous columns for X (0–18)
    cont_indices_y = slice(0, 12)   # continuous columns for Y (0–11)

    # Scale continuous features
    flat_x = np.vstack([x[:, cont_indices_x] for x in X])
    flat_y = np.vstack([y[:, cont_indices_y] for y in Y])

    scaler_X = MinMaxScaler().fit(flat_x)
    # scaler_Y = MinMaxScaler().fit(flat_y)
    scaler_Y = StandardScaler().fit(flat_y)
    return scaler_X, scaler_Y


def scale_input(X, Y, scaler_X, scaler_Y):
    cont_indices_x = slice(0, 19)   # continuous columns for X
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

def create_window_pred(sequence_x, window_size, n_steps_ahead=5):
    X, Y = [], []
    curr_len = len(sequence_x)

    # for t in range(1, len(sequence_x)):
    start = max(0, curr_len - window_size)
    window = sequence_x[start:curr_len]

    # padding na początku, jeśli okno krótsze niż window_size
    pad_len = window_size - len(window)
    if pad_len > 0:
        window = np.vstack([np.zeros((pad_len, sequence_x.shape[1])), window])
    X.append(window)

        

    return np.array(X)


def generate_predictions(model, input_seq, n_steps_ahead=5):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)  # add batch dimension
        predictions = model(input_tensor)
        return predictions.squeeze(0).numpy()  # remove batch dimension