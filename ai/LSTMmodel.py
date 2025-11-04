import joblib
from sklearn.preprocessing import StandardScaler , MinMaxScaler, RobustScaler
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
    def __init__(self, input_size, hidden_size, output_size, num_layers=1,dropout_prob=0.3):
        super(LSTMStatePredictor, self).__init__()
        
        # Zapisz parametry (potrzebne do ewentualnego ręcznego
        # tworzenia stanu, choć nie jest to już wymagane w forward)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        lstm_dropout_prob = dropout_prob if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,dropout=lstm_dropout_prob)

        self.dropout_layer = nn.Dropout(dropout_prob)
    
        # Twój świetny pomysł z wieloma głowicami - zostaje!
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, 2),  # progress
            nn.Linear(hidden_size, 1),  # fuel
            nn.Linear(hidden_size, 4),  # wear
            nn.Linear(hidden_size, 4),  # temp
            nn.Linear(hidden_size, 1)   # track wetness
        ])

        # Te pola nie są już potrzebne Wewnątrz modelu
        # self.scaler_X = None
        # self.scaler_Y = None
        # self.n_steps_ahead = n_steps_ahead # Usuwamy n_steps_ahead
        self.output_size = output_size

    def forward(self, x, h_c=None):
        # 1. Nie musisz ręcznie inicjować h_c. 
        #    nn.LSTM zrobi to automatycznie, jeśli h_c jest None.
        
        # 2. Przetwórz CAŁĄ sekwencję
        #    x ma kształt: [B, seq_len, 37]
        #    out będzie miał kształt: [B, seq_len, hidden_size]
        out, h_c = self.lstm(x, h_c) 

        out = self.dropout_layer(out)  # Zastosuj dropout do wyjścia LSTM
        
        # 3. Zastosuj głowice do CAŁEGO tensora 'out', a nie tylko 'out[:, -1, :]'
        #    head(out) da np. [B, seq_len, 2]
        outputs = [head(out) for head in self.heads]
        
        # 4. Połącz wzdłuż ostatniego wymiaru (wymiaru cech)
        #    List of [B,S,2], [B,S,1], [B,S,4]... -> [B, S, 12]
        #    Używamy dim=2, ponieważ kształt to (Batch, Seq_len, Features)
        combined = torch.cat(outputs, dim=2) 
        
        return combined, h_c

def create_scalers(X,Y):

    cont_indices_x = slice(0, 20)   # continuous columns for X (0–19)
    cont_indices_y = slice(0, 12)   # continuous columns for Y (0–11)

    # Scale continuous features
    flat_x = np.vstack([x[:, cont_indices_x] for x in X])
    flat_y = np.vstack([y[:, cont_indices_y] for y in Y])

    # scaler_X = MinMaxScaler().fit(flat_x)
    # # scaler_Y = MinMaxScaler().fit(flat_y)
    # scaler_Y = StandardScaler().fit(flat_y)
    scaler_X = RobustScaler().fit(flat_x)
    scaler_Y = RobustScaler().fit(flat_y)
    return scaler_X, scaler_Y

def scale_input(X, Y, scaler_X, scaler_Y):
    cont_indices_x = slice(0, 20)   # continuous columns for X
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

# def scale_single_input(x, scaler_X):
#     cont_indices_x = slice(0, 19)   # continuous columns for X
#     X_scaled_grouped = []

#     for x_seq in x:
#         x_scaled = np.array(x_seq, dtype=float)
#         x_scaled[:, cont_indices_x] = scaler_X.transform(x_seq[:, cont_indices_x])
#         X_scaled_grouped.append(x_scaled)
#     return X_scaled_grouped

def scale_single_input(raw_vector_x, scaler_x_cont):
    """
    Skaluje pojedynczy wektor (37,), stosując scaler tylko do 
    części ciągłej (0-19) i zostawiając kategorialną (20-36).
    """
    cont_indices_x = slice(0, 20)
    cat_indices_x = slice(20, 38)
    
    # raw_vector_x[cont_indices_x] ma kształt (19,)
    # Musimy go przekształcić na (1, 19) dla scalera
    x_cont_scaled = scaler_x_cont.transform([raw_vector_x[cont_indices_x]])
    
    # raw_vector_x[cat_indices_x] ma kształt (18,)
    # --- POPRAWKA TUTAJ ---
    # Musimy go przekształcić na (1, 18), aby pasował do hstack
    x_cat = raw_vector_x[cat_indices_x].reshape(1, -1)
    
    # Teraz łączymy (1, 19) z (1, 18) -> (1, 37)
    # i spłaszczamy z powrotem do 1D (37,)
    return np.hstack([x_cont_scaled, x_cat]).flatten()
       

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



def generate_predictions(model, input_seq,scaler_X=None, scaler_Y=None,h_c=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    lap_dist_sin = np.sin(2 * np.pi * input_seq[0])

    lap_dist_cos = np.cos(2 * np.pi * input_seq[0])
    # input_seq = create_window_pred(input_seq, window_size=30, n_steps_ahead=n_steps_ahead)
    input_seq = np.hstack([
        lap_dist_sin, 
        lap_dist_cos, 
        input_seq[1:]  # <-- Pomijamy starą cechę LAP_DIST
    ])
    input_seq = scale_single_input(input_seq, scaler_X)



    
         
    with torch.no_grad():
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).reshape(1, 1, 38).to(device)
        predictions , h_c = model(input_tensor, h_c)
        predictions = predictions.cpu().numpy().reshape(1, 13)
        
        predictions = scaler_Y.inverse_transform(predictions)
        return predictions.flatten(), h_c
    
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
            Y_seq.append(race[i + 1][:-28]) 
        
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

def create_sliding_windows(races_x_list, races_y_list, sequence_length, step=1):
    """
    Tworzy próbki (X, Y) metodą przesuwnego okna dla Teacher Forcing.
    X = [t, t+1, ..., t+seq_len-1]
    Y = [t+1, t+2, ..., t+seq_len]  (przesunięte o 1)
    """
    all_X_samples = []
    all_Y_samples = []
    
    # Pamiętaj, że race_y to już wyodrębnione 12 cech
    # race_x to pełne 37 cech
    
    for race_x, race_y in zip(races_x_list, races_y_list):
        race_length = race_x.shape[0]
        
        # Pętla po pojedynczym wyścigu
        # Ostatni indeks startowy `i` musi być taki, aby `i + sequence_length`
        # nie wyszło poza zakres dla Y (który jest przesunięty o 1)
        for i in range(0, race_length - sequence_length, step):
            
            # X: Kształt (sequence_length, 37)
            x_sample = race_x[i : i + sequence_length]
            
            # Y: Kształt (sequence_length, 12)
            # Dla wejścia X w kroku 't', celem jest Y z kroku 't+1'
            y_sample = race_y[i + 1 : i + sequence_length + 1] 
            
            all_X_samples.append(x_sample)
            all_Y_samples.append(y_sample)
            
    return np.array(all_X_samples), np.array(all_Y_samples)
    
def train_model():
    data = load_data_from_db()
    
   
    X, Y = create_x_y(data)
    input_size = X[0].shape[1]
    output_size = Y[0].shape[1]
    print(input_size, output_size)

    SEQUENCE_LENGTH = 800
    STEP = 1

    lr = 1e-4
    batch_size = 128
    num_epochs = 40
    weight = [0.5, 1.8, 3.0, 0.1, 1.5]
   

    scaler_X, scaler_Y = create_scalers(X,Y)

    X_train, Y_train = scale_input(X,Y,scaler_X,scaler_Y)
    
    # n_steps_ahead = 5  # number of future steps to predict


    # all_X, all_Y = [], []
    # for race_x, race_y in zip(X_train, Y_train):  
    #     X_r, Y_r = create_windows(race_x, race_y, window_size=30, n_steps_ahead=n_steps_ahead)
    #     all_X.append(X_r)
    #     all_Y.append(Y_r)

    print("Tworzenie sampli treningowych...")
    X_train_samples, Y_train_samples = create_sliding_windows(
        X_train, Y_train, SEQUENCE_LENGTH, STEP
    )

    # X_train = np.vstack(all_X)  # shape: [N_samples, window_size, n_features]
    # Y_train = np.vstack(all_Y) 
    # all_X, all_Y = [], []
    # for race_x, race_y in zip(X_test, Y_test):  
    #     X_r, Y_r = create_windows(race_x, race_y, window_size=30)
    #     all_X.append(X_r)
    #     all_Y.append(Y_r)
    # X_test = np.vstack(all_X)  # shape: [N_samples, window_size, n_features]
    # Y_test = np.vstack(all_Y)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = LSTMStatePredictor(input_size=input_size, hidden_size=256, output_size=output_size, num_layers=1).to(device)



    
    
    X_train_tensor = torch.tensor(X_train_samples, dtype=torch.float32).to(device)
    Y_train_tensor = torch.tensor(Y_train_samples, dtype=torch.float32).to(device)
    

    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # optimizer = optim.Adam(model.parameters(), lr=lr)
    

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5)
    loss_cont = nn.MSELoss()
    
    for epoch in range(num_epochs):
        
        model.train()
        total_train_loss = 0

      
        

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # Model dostaje całą sekwencję 200 kroków
            # i zwraca predykcje dla całej sekwencji 200 kroków
            y_pred, _ = model(x_batch) 
            
            # y_pred ma kształt (batch_size, SEQUENCE_LENGTH, 12)
            
            # Obliczamy stratę dla całej sekwencji na raz
            # Musimy indeksować wymiar cech [:, :, ...]
            loss_progress = loss_cont(y_pred[:, :, 0:2], y_batch[:, :, 0:2])
            loss_fuel     = loss_cont(y_pred[:, :, 2:3], y_batch[:, :, 2:3])
            loss_wear     = loss_cont(y_pred[:, :, 3:7], y_batch[:, :, 3:7])
            loss_temp     = loss_cont(y_pred[:, :, 7:11], y_batch[:, :, 7:11])
            loss_wet      = loss_cont(y_pred[:, :, 11:], y_batch[:, :, 11:])
            
            # Sumujemy straty (tak jak miałeś)
            loss = (weight[0] * loss_progress + 
                    weight[1] * loss_fuel + 
                    weight[2] * loss_wear + 
                    weight[3] * loss_temp + 
                    weight[4] * loss_wet)
            
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        scheduler.step(avg_train_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")
    torch.save(model.state_dict(), "models/lstm1_model.pth")
    import joblib
    joblib.dump(scaler_X, "models/scaler1_X.pkl")
    joblib.dump(scaler_Y, "models/scaler1_Y.pkl")

    print("✅ Model saved to models/lstm1_model.pth")

train_model()