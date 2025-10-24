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
    def __init__(self, input_size, hidden_size, output_size,n_steps_ahead, num_layers=1,scaler_X=None, scaler_Y=None):
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

        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
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

def scale_single_input(x, scaler_X):
    cont_indices_x = slice(0, 19)   # continuous columns for X
    X_scaled_grouped = []

    for x_seq in x:
        x_scaled = np.array(x_seq, dtype=float)
        x_scaled[:, cont_indices_x] = scaler_X.transform(x_seq[:, cont_indices_x])
        X_scaled_grouped.append(x_scaled)
    return X_scaled_grouped
       

def create_window_pred(sequence_x, window_size, n_steps_ahead=5):
    X, Y = [], []
    sequence_x = np.array(sequence_x)
    curr_len = len(sequence_x)

    start = max(0, curr_len - window_size)
    window = sequence_x[start:curr_len]

    pad_len = window_size - len(window)
    if pad_len > 0:
        window = np.vstack([np.zeros((pad_len, sequence_x.shape[1])), window])

    # dodaj batch dimension
    X = window[np.newaxis, :, :]  # shape [1, window_size, num_features]
    return X



def generate_predictions(model, input_seq, n_steps_ahead=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    input_seq = create_window_pred(input_seq, window_size=30, n_steps_ahead=n_steps_ahead)
    input_seq = scale_single_input(input_seq, model.scaler_X)

    
         
    with torch.no_grad():
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).to(device)
        predictions = model(input_tensor)
        predictions = predictions.detach().cpu().numpy()
        b, t, f = predictions.shape
        predictions = predictions.reshape(b * t, f)
        predictions = model.scaler_Y.inverse_transform(predictions)
        return predictions  
    
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


def create_x_y(data):
    X_grouped, Y_grouped = [], []

    for race in data:
        X_seq, Y_seq = [], []
        for i in range(len(race) - 1):
            X_seq.append(race[i][:-2])
            Y_seq.append(race[i + 1][:-27]) 
        
        # dodajemy każdy wyścig osobno
        X_grouped.append(np.array(X_seq, dtype=float))
        Y_grouped.append(np.array(Y_seq, dtype=float))

    return X_grouped, Y_grouped

def create_windows(sequence_x, sequence_y, window_size, n_steps_ahead=5):
    X, Y = [], []
    for t in range(1, len(sequence_x)):
        start = max(0, t - window_size)
        window = sequence_x[start:t]

        # padding na początku, jeśli okno krótsze niż window_size
        pad_len = window_size - len(window)
        if pad_len > 0:
            window = np.vstack([np.zeros((pad_len, sequence_x.shape[1])), window])
        X.append(window)

        # Y: wypełniamy zerami, jeśli końcówka wyścigu ma mniej niż n_steps_ahead
        y_window = sequence_y[t:t+n_steps_ahead]
        if y_window.shape[0] < n_steps_ahead:
            pad = np.zeros((n_steps_ahead - y_window.shape[0], sequence_y.shape[1]))
            y_window = np.vstack([y_window, pad])
        Y.append(y_window)

    return np.array(X), np.array(Y)
    
def train_model():
    data = load_data_from_db()
    
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

    X, Y = create_x_y(data)
    input_size = X[0].shape[1]
    output_size = Y[0].shape[1]
    print(input_size, output_size)


    lr = 1e-3
    batch_size = 128
    num_epochs = 75
    weight = [0.5, 1.8, 3.0, 0.1, 1.5]
   

    scaler_X, scaler_Y = create_scalers(X,Y)

    X_train, Y_train = scale_input(X,Y,scaler_X,scaler_Y)
    
    n_steps_ahead = 5  # number of future steps to predict


    all_X, all_Y = [], []
    for race_x, race_y in zip(X_train, Y_train):  
        X_r, Y_r = create_windows(race_x, race_y, window_size=30, n_steps_ahead=n_steps_ahead)
        all_X.append(X_r)
        all_Y.append(Y_r)

    X_train = np.vstack(all_X)  # shape: [N_samples, window_size, n_features]
    Y_train = np.vstack(all_Y) 
    # all_X, all_Y = [], []
    # for race_x, race_y in zip(X_test, Y_test):  
    #     X_r, Y_r = create_windows(race_x, race_y, window_size=30)
    #     all_X.append(X_r)
    #     all_Y.append(Y_r)
    # X_test = np.vstack(all_X)  # shape: [N_samples, window_size, n_features]
    # Y_test = np.vstack(all_Y)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = LSTMStatePredictor(input_size=input_size, hidden_size=128, output_size=output_size,n_steps_ahead=n_steps_ahead, num_layers=1).to(device)



    
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).to(device)
    

    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # optimizer = optim.Adam(model.parameters(), lr=lr)
    

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5)
    loss_cont = nn.MSELoss()
    

    
    
    
    for epoch in range(num_epochs):
        
        model.train()
        total_loss = 0


        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(x_batch)
            
            # bierzemy ostatni krok z predykcji (lub całość, jeśli tak trenujesz)
            pred_flat = pred[:, -1, :]
            y_flat = y_batch[:, -1, :]
            # rozbijanie po grupach
            loss_progress = loss_cont(pred_flat[:, 0:2], y_flat[:, 0:2])
            loss_fuel     = loss_cont(pred_flat[:, 2:3], y_flat[:, 2:3])
            loss_wear     = loss_cont(pred_flat[:, 3:7], y_flat[:, 3:7])
            loss_temp     = loss_cont(pred_flat[:, 7:11], y_flat[:, 7:11])
            loss_wet      = loss_cont(pred_flat[:, 11:], y_flat[:, 11:])
            # łączymy straty z różnych grup z różnymi wagami
            loss =  weight[0] * loss_progress + \
                    weight[1] * loss_fuel + \
                    weight[2] * loss_wear + \
                    weight[3] * loss_temp + \
                    weight[4] * loss_wet
           
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step(total_loss)

    torch.save(model.state_dict(), "models/lstm_model.pth")
    import joblib
    joblib.dump(scaler_X, "models/scaler_X.pkl")
    joblib.dump(scaler_Y, "models/scaler_Y.pkl")

    print("✅ Model saved to models/lstm_model.pth")

# train_model()