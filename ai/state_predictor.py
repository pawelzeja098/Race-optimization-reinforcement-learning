import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
# from envs.filtr_json_from_race import load_from_db
import sqlite3
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, f1_score


class RaceModel(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU()
        )
        self.cont_head = nn.Linear(128,output_size)             # ciągłe
      
        self.scaler_X = None
        self.scaler_Y = None

    def forward(self, x):
        h = self.shared(x)
        cont = self.cont_head(h)
        # cats = [head(h) for head in self.cat_heads]
        return cont#, cats
    
    def scale_input(self, X_grouped,Y_grouped):
         # --- Continous and discrete feature indices ---
        cont_indices_x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]     # continuous
        cont_indices_y = [0,1,2,3,4,5,6,7,8,9,10,11]          # continuous
     

        # --- Scale continuous features ---
        all_X_cont = np.vstack([x[:, cont_indices_x] for x in X_grouped])
        all_Y_cont = np.vstack([y[:, cont_indices_y] for y in Y_grouped])
        self.scaler_X = StandardScaler().fit(all_X_cont)
        self.scaler_Y = StandardScaler().fit(all_Y_cont)

        # Scale X and Y for continuous features
        X_scaled_grouped = []
        Y_scaled_grouped = []
        for x_seq, y_seq in zip(X_grouped, Y_grouped):
            x_scaled = np.array(x_seq)
            x_scaled[:, cont_indices_x] = self.scaler_X.transform(x_seq[:, cont_indices_x])
            X_scaled_grouped.append(x_scaled)

            y_scaled = np.array(y_seq)
            y_scaled[:, cont_indices_y] = self.scaler_Y.transform(y_seq[:, cont_indices_y])
            Y_scaled_grouped.append(y_scaled)

        # Conversion to torch tensors
        X_t = [torch.tensor(x, dtype=torch.float32) for x in X_scaled_grouped]
        Y_cont_t = [torch.tensor(y[:, cont_indices_y], dtype=torch.float32) for y in Y_scaled_grouped]
      

        return X_t, Y_cont_t
    
    
    
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

    X_grouped, Y_grouped = [], []

    for race_id, states_json in rows:
        states = json.loads(states_json)

        X_seq, Y_seq = [], []
        for i in range(len(states) - 1):
            X_seq.append(states[i][:-2]) # bez ostatnich 2 kolumn (ostatnie okrazenie, najlepsze okrazenie)
            Y_seq.append(states[i + 1][:-24])  # bez ostatnich 24 kolumn (pogoda,uszkodzenia itp.)

        X_grouped.append(np.array(X_seq, dtype=float))
        Y_grouped.append(np.array(Y_seq, dtype=float))

    return X_grouped, Y_grouped
    
if __name__ == "__main__":
    X, Y = load_data_from_db()

    input_size = X[0].shape[1]
    output_size = Y[0].shape[1]
    loo = LeaveOneOut()
    all_fold_train = []
    all_fold_test = []
    for fold, (train_idx, test_idx) in enumerate(loo.split(X)):
        

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
        model = RaceModel(input_size, output_size).to(device)

        X, Y_cont = model.scale_input(X, Y)

        optimizer = optim.Adam(model.parameters(), lr=5e-5)
        loss_cont = nn.MSELoss()
        loss_cat  = nn.CrossEntropyLoss()

        fold_train_losses = []
        fold_test_losses = []

        print(f"Fold {fold+1}")
        X_train = [X[i] for i in train_idx]
        X_test  = [X[i] for i in test_idx]
        Y_cont_train = [Y_cont[i] for i in train_idx]
        Y_cont_test  = [Y_cont[i] for i in test_idx]
        # Y_cat_train  = [Y_cat[i] for i in train_idx]
        # Y_cat_test   = [Y_cat[i] for i in test_idx]


    # X_train, Y_cont_train, Y_cat_train = X[:2], Y_cont[:2], Y_cat[:2]  # dwa pierwsze wyścigi do treningu
    # X_test, Y_cont_test, Y_cat_test   = X[2:], Y_cont[2:], Y_cat[2:]

        X_train = [x.to(device) for x in X_train]
        Y_cont_train = [y.to(device) for y in Y_cont_train]
        # Y_cat_train = [[y.to(device) for y in y_seq] for y_seq in Y_cat_train]

        X_test = [x.to(device) for x in X_test]
        Y_cont_test = [y.to(device) for y in Y_cont_test]   
        # Y_cat_test = [[y.to(device) for y in y_seq] for y_seq in Y_cat_test]
        
        n_epochs = 750
        for epoch in range(n_epochs):
            total_loss = 0.0
            model.train()
            for x_seq, y_cont_seq in zip(X_train, Y_cont_train): # y_cat_seq, Y_cat_train):
                # print(x_seq, y_cont_seq, y_cat_seq)
                optimizer.zero_grad()

                cont_pred = model(x_seq) # continuous and list of categorical predictions , cat_preds

                loss = loss_cont(cont_pred, y_cont_seq)
                # loss_k = sum(loss_cat(cat_pred, y_cat) for cat_pred, y_cat in zip(cat_preds, y_cat_seq))
                # loss = loss_c + loss_k
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(X_train)

            total_test_loss = 0.0
            model.eval()
            all_mse = []


            with torch.no_grad():
                for x_seq, y_cont_seq in zip(X_test, Y_cont_test): #,y_cat_seq, Y_cat_test):
                    x_seq = x_seq.to(device)
                    y_cont_seq = y_cont_seq.to(device)
                    # y_cat_seq = [y.to(device) for y in y_cat_seq]

                    cont_pred = model(x_seq) #, cat_preds = model(x_seq)

                    loss = loss_cont(cont_pred, y_cont_seq)

                    total_test_loss += loss.item()

            avg_test_loss = total_test_loss / len(X_test)
            # print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
            
            fold_train_losses.append(avg_train_loss)
            fold_test_losses.append(avg_test_loss)
        
        all_fold_train.append(fold_train_losses)
        all_fold_test.append(fold_test_losses)

        # plt.plot(fold_train_losses, label=f'Fold {fold+1} Train', linestyle='-')
        # plt.plot(fold_test_losses,  label=f'Fold {fold+1} Test',  linestyle='--')

    for fold in range(len(all_fold_train)):
        plt.figure()
        plt.plot(all_fold_train[fold], label='Train')
        plt.plot(all_fold_test[fold], label='Test')
        plt.title(f'Fold {fold+1}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
   