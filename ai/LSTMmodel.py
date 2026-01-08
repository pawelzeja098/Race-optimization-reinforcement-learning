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
from config import X_SHAPE, Y_SHAPE, CONT_LENGTH, CAT_LENGTH, Y_INDEXES, NO_SCALER_IDXES_X, MIN_MAX_SCALER_IDXES_X, ROBUST_SCALER_IDXES_X, NO_SCALER_IDEXES_Y, MIN_MAX_SCALER_IDXES_Y, ROBUST_SCALER_IDXES_Y



class LSTMStatePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1,dropout_prob=0.3):
        super(LSTMStatePredictor, self).__init__()
        
        

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        lstm_dropout_prob = dropout_prob if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,dropout=lstm_dropout_prob)

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.act_delta = nn.ReLU()
        self.act_pos = nn.Tanh()
    
      
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, 2),  # progress
            nn.Linear(hidden_size, 1),  # fuel
            nn.Linear(hidden_size, 4),  # wear
            nn.Linear(hidden_size, 4) # temp
            # nn.Linear(hidden_size, 1)   # track wetness
        ])

      
        self.output_size = output_size

    def forward(self, x, h_c=None):
        # 1. Nie trzeba ręcznie inicjować h_c. 
        #    nn.LSTM zrobi to automatycznie, jeśli h_c jest None.
        
  
        out, h_c = self.lstm(x, h_c) 

        out = self.dropout_layer(out)  # Zastosuj dropout do wyjścia LSTM
        
        # 3. Przetwórz każde wyjście przez odpowiednią głowicę
        outputs = [head(out) for head in self.heads]

        #4. Aktywacje specyficzne dla każdej głowicy
        outputs[0] = self.act_pos(outputs[0])
        outputs[1] = self.act_delta(outputs[1])
        outputs[2] = self.act_delta(outputs[2])
        


        
        # 4. Połącz wzdłuż ostatniego wymiaru (wymiaru cech)
        #    List of [B,S,2], [B,S,1], [B,S,4]... -> [B, S, 12]
        #    Używamy dim=2, ponieważ kształt to (Batch, Seq_len, Features)
        combined = torch.cat(outputs, dim=2) 
        
        return combined, h_c

def create_scalers(X,Y):

    # cont_indices_x = slice(0, CONT_LENGTH)   # continuous columns for X (0–18)
    # cont_indices_y = slice(0, Y_SHAPE)   # continuous columns for Y (0–11)

    no_scaler_x = slice(0, NO_SCALER_IDXES_X)  # no scaler for X
    min_max_scaler_x = slice(NO_SCALER_IDXES_X, MIN_MAX_SCALER_IDXES_X)  # min-max scaler for X
    robust_scaler_x = slice(MIN_MAX_SCALER_IDXES_X, ROBUST_SCALER_IDXES_X)  # robust scaler for X
    no_scaler_y = slice(0, NO_SCALER_IDEXES_Y)  # no scaler for Y
    min_max_scaler_y = slice(NO_SCALER_IDEXES_Y, MIN_MAX_SCALER_IDXES_Y)  # min-max scaler for Y
    robust_scaler_y = slice(MIN_MAX_SCALER_IDXES_Y, ROBUST_SCALER_IDXES_Y)  # robust scaler for Y

   
    flat_x_min_max = np.vstack([x[:, min_max_scaler_x] for x in X])
    flat_x_robust = np.vstack([x[:, robust_scaler_x] for x in X])
    flat_y_min_max = np.vstack([y[:, min_max_scaler_y] for y in Y])
    flat_y_robust = np.vstack([y[:, robust_scaler_y] for y in Y])

    scaler_X_min_max = MinMaxScaler().fit(flat_x_min_max)
    scaler_X_robust = RobustScaler().fit(flat_x_robust)
    scaler_Y_min_max = MinMaxScaler().fit(flat_y_min_max)
    scaler_Y_robust = RobustScaler().fit(flat_y_robust)

   
    return scaler_X_min_max, scaler_X_robust, scaler_Y_min_max, scaler_Y_robust

def scale_input(X, Y, scaler_X_min_max, scaler_X_robust, scaler_Y_min_max, scaler_Y_robust):
    no_scaler_x = slice(0, NO_SCALER_IDXES_X)  # no scaler for X
    min_max_scaler_x = slice(NO_SCALER_IDXES_X, MIN_MAX_SCALER_IDXES_X)  # min-max scaler for X
    robust_scaler_x = slice(MIN_MAX_SCALER_IDXES_X, ROBUST_SCALER_IDXES_X)  # robust scaler for X
    no_scaler_y = slice(0, NO_SCALER_IDEXES_Y)  # no scaler for Y
    min_max_scaler_y = slice(NO_SCALER_IDEXES_Y, MIN_MAX_SCALER_IDXES_Y)  # min-max scaler for Y
    robust_scaler_y = slice(MIN_MAX_SCALER_IDXES_Y, ROBUST_SCALER_IDXES_Y)  # robust scaler for Y

    X_scaled_grouped = []
    Y_scaled_grouped = []

    for x_seq, y_seq in zip(X, Y):
        x_scaled = np.array(x_seq, dtype=float)
        x_scaled[:, min_max_scaler_x] = scaler_X_min_max.transform(x_seq[:, min_max_scaler_x])
        x_scaled[:, robust_scaler_x] = scaler_X_robust.transform(x_seq[:, robust_scaler_x])
        X_scaled_grouped.append(x_scaled)

        y_scaled = np.array(y_seq, dtype=float)
        y_scaled[:, min_max_scaler_y] = scaler_Y_min_max.transform(y_seq[:, min_max_scaler_y])
        y_scaled[:, robust_scaler_y] = scaler_Y_robust.transform(y_seq[:, robust_scaler_y])
        Y_scaled_grouped.append(y_scaled)

    # Conversion to torch tensors
    # X_t = [torch.tensor(x, dtype=torch.float32) for x in X_scaled_grouped]
    # Y_cont_t = [torch.tensor(y[:, cont_indices_y], dtype=torch.float32) for y in Y_scaled_grouped]

    return X_scaled_grouped, Y_scaled_grouped


def scale_single_input(raw_vector_x, scaler_X_min_max, scaler_X_robust):
    """
    Skaluje pojedynczy wektor (37,), stosując scaler tylko do 
    części ciągłej (0-19) i zostawiając kategorialną (20-36).
    """
    no_scaler_x = slice(0, NO_SCALER_IDXES_X)  # no scaler for X
    min_max_scaler_x = slice(NO_SCALER_IDXES_X, MIN_MAX_SCALER_IDXES_X)  # min-max scaler for X
    robust_scaler_x = slice(MIN_MAX_SCALER_IDXES_X, ROBUST_SCALER_IDXES_X)  # robust scaler for X
 
    
    # raw_vector_x[cont_indices_x] ma kształt (19,)

    x_min_max_scaled = scaler_X_min_max.transform([raw_vector_x[min_max_scaler_x]])
    x_robust_scaled = scaler_X_robust.transform([raw_vector_x[robust_scaler_x]])
    
    # raw_vector_x[cat_indices_x] ma kształt (18,)

    # Musimy go przekształcić na (1, 19), aby pasował do hstack
    x_no_scaled = raw_vector_x[no_scaler_x].reshape(1, -1)
    
  
    return np.hstack([x_no_scaled, x_min_max_scaled, x_robust_scaled]).flatten()
       


def generate_predictions(model, input_seq,scaler_X_min_max=None, scaler_X_robust=None, scaler_Y_min_max=None, scaler_Y_robust=None,h_c=None):
    no_scaler_y = slice(0, NO_SCALER_IDEXES_Y)  # no scaler for Y
    min_max_scaler_y = slice(NO_SCALER_IDEXES_Y, MIN_MAX_SCALER_IDXES_Y)  # min-max scaler for Y
    robust_scaler_y = slice(MIN_MAX_SCALER_IDXES_Y, ROBUST_SCALER_IDXES_Y)
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
    input_seq = scale_single_input(input_seq, scaler_X_min_max, scaler_X_robust)



    
         
    with torch.no_grad():
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).reshape(1, 1, X_SHAPE).to(device)
        predictions , h_c = model(input_tensor, h_c)

        # predictions_scaled = predictions.reshape(1, Y_SHAPE)
        # predictions_raw = predictions_scaled.cpu().numpy().flatten()

        predictions_scaled = predictions.cpu().numpy().reshape(1, Y_SHAPE)

        predictions_raw = np.zeros_like(predictions_scaled)
        
        # A. No Scaler (Sin/Cos) - Przepisujemy
        predictions_raw[:, no_scaler_y] = predictions_scaled[:, no_scaler_y]
        
        # B. MinMax (Delty) - Odwracamy
  
        predictions_raw[:, min_max_scaler_y] = scaler_Y_min_max.inverse_transform(predictions_scaled[:, min_max_scaler_y])
        
        # C. Robust (Temperatury) - Odwracamy
     
        predictions_raw[:, robust_scaler_y] = scaler_Y_robust.inverse_transform(predictions_scaled[:, robust_scaler_y])
        
        h_c = (h_c[0].detach(), h_c[1].detach())  # Odłączamy stany od grafu obliczeń

        return predictions_raw.flatten(), h_c
    
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
            X_seq.append(race[i][:Y_INDEXES])  # current state
            Y_seq.append(race[i + 1][Y_INDEXES:])  # next state
        # dodajemy każdy wyścig osobno
        X_grouped.append(np.array(X_seq, dtype=float))
        Y_grouped.append(np.array(Y_seq, dtype=float))

    return X_grouped, Y_grouped

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
    num_epochs = 20
    weight = [1.8, 0.7, 1.3, 0.2]
   

    scaler_X_min_max, scaler_X_robust, scaler_Y_min_max, scaler_Y_robust = create_scalers(X,Y)

    X_train, Y_train = scale_input(X,Y,scaler_X_min_max, scaler_X_robust, scaler_Y_min_max, scaler_Y_robust)
    
   
    print("Tworzenie sampli treningowych...")
    X_train_samples, Y_train_samples = create_sliding_windows(
        X_train, Y_train, SEQUENCE_LENGTH, STEP
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = LSTMStatePredictor(input_size=input_size, hidden_size=256, output_size=output_size, num_layers=1).to(device)

    
    X_train_tensor = torch.tensor(X_train_samples, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train_samples, dtype=torch.float32)
    

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
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            # Model dostaje całą sekwencję 200 kroków
            # i zwraca predykcje dla całej sekwencji 200 kroków
            y_pred, _ = model(x_batch) 
            
            # y_pred ma kształt (batch_size, SEQUENCE_LENGTH, 12)
            
            # Obliczamy stratę dla całej sekwencji na raz
    
            loss_progress = loss_cont(y_pred[:, :, 0:2], y_batch[:, :, 0:2])
            loss_fuel     = loss_cont(y_pred[:, :, 2:3], y_batch[:, :, 2:3])
            loss_wear     = loss_cont(y_pred[:, :, 3:7], y_batch[:, :, 3:7])
            loss_temp     = loss_cont(y_pred[:, :, 7:11], y_batch[:, :, 7:11])

            
            # Sumujemy straty (tak jak miałeś)
            loss = (weight[0] * loss_progress + 
                    weight[1] * loss_fuel + 
                    weight[2] * loss_wear + 
                    weight[3] * loss_temp 
                    # weight[4] * loss_wet)
            )
            
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        scheduler.step(avg_train_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")
    torch.save(model.state_dict(), "models/lstmdeltaT_model1.pth")
    import joblib
    joblib.dump(scaler_X_min_max, "models/scalerX_min_max1.pkl")
    joblib.dump(scaler_X_robust, "models/scalerX_robust1.pkl")
    joblib.dump(scaler_Y_min_max, "models/scalerY_min_max1.pkl")
    joblib.dump(scaler_Y_robust, "models/scalerY_robust1.pkl")

    print("✅ Model saved to models/lstmdeltaT_model1.pth")

# train_model()


# import torch
# from torchview import draw_graph
# import graphviz

# # 1. Konfiguracja modelu
# model = LSTMStatePredictor(input_size=37, hidden_size=256, output_size=11, num_layers=1)

# # 2. Przygotowanie danych (X oraz Stan)
# dummy_x = torch.randn(1, 1, 37)
# # Stan dla LSTM: (h_0, c_0) -> rozmiar [layers, batch, hidden]
# dummy_h = torch.randn(1, 1, 256)
# dummy_c = torch.randn(1, 1, 256)
# dummy_state = (dummy_h, dummy_c)

# # 3. Generowanie grafu
# model_graph = draw_graph(
#     model, 
#     input_data=(dummy_x, dummy_state),  # Przekazujemy X i Stan
#     depth=1, 
#     expand_nested=True,
#     save_graph=False  # Nie zapisuj jeszcze automatycznie
# )

# # Zamiast render(), wypisz źródło:
# print(model_graph.visual_graph.source)

