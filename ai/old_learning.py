data = load_data_from_db()
# input_size = X[0].shape[1]
# output_size = Y[0].shape[1]
# random.seed(42)
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

# scaler_X, scaler_Y = create_scalers(X,Y)

# X_scaled, Y_scaled = scale_input(X,Y,scaler_X,scaler_Y)

# X_scaled = [scaler_X.transform(np.array(race)) for race in X]
# Y_scaled = [scaler_Y.transform(np.array(race)) for race in Y]



loo = LeaveOneOut()
all_fold_train = []
all_fold_test = []
all_fold_r2_test = []
i = 0

# lrs = [1e-3,1e-3,1e-3, 7e-4, 5e-4, 3e-4, 1e-4, 7e-5, 5e-5, 3e-5, 1e-5]
lr = 1e-3
# lr = 7e-5
# lr = 5e-4
# batch_sizes = [32, 64, 128, 256, 512]
batch_size = 128
# num_epochs = [50, 100, 150, 200, 300, 400]
num_epochs = 75
weights = [
    # [1.0, 1.0, 1.0, 1.0, 1.0],
    # [0.8, 1.2, 1.5, 0.5, 1.0],
    [0.7, 1.4, 2.0, 0.3, 1.2],
    [0.6, 1.6, 2.5, 0.2, 1.3],
    [0.5, 1.8, 3.0, 0.1, 1.5],
    [1.0,1.0,2.5,0.1,0.4]
]
weight = [0.5, 1.8, 3.0, 0.1, 1.5]
# weight = weights[4]  # wybierz zestaw wag do testowania
# for weight in weights:

print("Testing weight:", weight)
for fold, (train_idx, test_idx) in enumerate(loo.split(X)):
    
    if fold > 0:
        break
    i += 1


    X_train = [X[i] for i in train_idx]
    X_test  = [X[i] for i in test_idx]
    Y_train = [Y[i] for i in train_idx]
    Y_test  = [Y[i] for i in test_idx]

    Y_train_raw = [np.array(y, copy=True) for y in Y_train]  # zapisz oryginalne (nieprzeskalowane)


    # Y_train_raw = Y_train.copy()  # zachowaj surowe Y_train przed skalowaniem

    scaler_X, scaler_Y = create_scalers(X_train,Y_train)

    X_train, Y_train = scale_input(X_train,Y_train,scaler_X,scaler_Y)
    X_test, Y_test = scale_input(X_test,Y_test,scaler_X,scaler_Y)
    n_steps_ahead = 5  # number of future steps to predict


    all_X, all_Y = [], []
    for race_x, race_y in zip(X_train, Y_train):  
        X_r, Y_r = create_windows(race_x, race_y, window_size=30, n_steps_ahead=n_steps_ahead)
        all_X.append(X_r)
        all_Y.append(Y_r)

    X_train = np.vstack(all_X)  # shape: [N_samples, window_size, n_features]
    Y_train = np.vstack(all_Y) 
    all_X, all_Y = [], []
    for race_x, race_y in zip(X_test, Y_test):  
        X_r, Y_r = create_windows(race_x, race_y, window_size=30)
        all_X.append(X_r)
        all_Y.append(Y_r)
    X_test = np.vstack(all_X)  # shape: [N_samples, window_size, n_features]
    Y_test = np.vstack(all_Y)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = LSTMStatePredictor(input_size=input_size, hidden_size=128, output_size=output_size,n_steps_ahead=n_steps_ahead, num_layers=1).to(device)



    
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # optimizer = optim.Adam(model.parameters(), lr=lr)
    

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5)
    loss_cont = nn.MSELoss()
    loss_cat  = nn.CrossEntropyLoss()

    fold_train_losses = []
    fold_test_losses = []
    # num_epochs = 100
    all_r2_per_output = []
    fold_r2_scores = []

    
    
    for epoch in range(num_epochs):
        
        model.train()
        total_loss = 0

        # weights = torch.tensor(
        #     [0.8, 0.8, 2.0, 1.5, 1.5, 1.5, 1.5, 0.1, 0.1, 0.1, 0.1, 3.0],
        #     dtype=torch.float32,
        #     device=device
        # )

        # for x_batch, y_batch in train_loader:
        #     optimizer.zero_grad()
        #     pred = model(x_batch)
            
        #     # bierzemy ostatni krok z predykcji (lub całość, jeśli tak trenujesz)
        #     pred_flat = pred[:, -1, :]
        #     y_flat = y_batch[:, -1, :]
        #     # rozbijanie po grupach
        #     loss_progress = loss_cont(pred_flat[:, 0:2], y_flat[:, 0:2])
        #     loss_fuel     = loss_cont(pred_flat[:, 2:3], y_flat[:, 2:3])
        #     loss_wear     = loss_cont(pred_flat[:, 3:7], y_flat[:, 3:7])
        #     loss_temp     = loss_cont(pred_flat[:, 7:11], y_flat[:, 7:11])
        #     loss_wet      = loss_cont(pred_flat[:, 11:], y_flat[:, 11:])
        #     # łączymy straty z różnych grup z różnymi wagami
        #     loss =  weight[0] * loss_progress + \
        #             weight[1] * loss_fuel + \
        #             weight[2] * loss_wear + \
        #             weight[3] * loss_temp + \
        #             weight[4] * loss_wet
        #     # łączenie z wagami
        #     # loss = (0.8 * loss_progress +
        #     #         1.2 * loss_fuel +
        #     #         1.5 * loss_wear +
        #     #         0.5 * loss_temp +
        #     #         1.0 * loss_wet)

        #     # standardowy update
        #     loss.backward()
        #     optimizer.step()
        #     total_loss += loss.item()
        tf_prob = 0.7  # prawdopodobieństwo użycia prawdziwej wartości

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()

            batch_size, seq_len, _ = x_batch.shape
            h_c = None
            x_input = x_batch.clone()  # startowe okno 30 kroków
            loss = 0.0

            for step in range(n_steps_ahead):  # generujemy 5 kroków
                y_pred, h_c = model(x_input, h_c)  # predykcja 1 kroku
                y_true_step = y_batch[:, step, :]

                # rozbijamy na grupy i liczymy stratę
                loss_progress = loss_cont(y_pred[:, 0:2], y_true_step[:, 0:2])
                loss_fuel     = loss_cont(y_pred[:, 2:3], y_true_step[:, 2:3])
                loss_wear     = loss_cont(y_pred[:, 3:7], y_true_step[:, 3:7])
                loss_temp     = loss_cont(y_pred[:, 7:11], y_true_step[:, 7:11])
                loss_wet      = loss_cont(y_pred[:, 11:], y_true_step[:, 11:])
                step_loss = weight[0]*loss_progress + weight[1]*loss_fuel + weight[2]*loss_wear + weight[3]*loss_temp + weight[4]*loss_wet
                loss += step_loss

                # Teacher forcing: decydujemy co wstawimy do okna na kolejny krok
                use_teacher = torch.rand(batch_size, device=x_batch.device) < tf_prob
                use_teacher = use_teacher.unsqueeze(1).float()  # shape [B,1]
                y_next_input = use_teacher * y_true_step + (1 - use_teacher) * y_pred

                # przesuwamy okno o 1 krok (autoreg)
                x_input = torch.cat([x_input[:, 1:, :], y_next_input.unsqueeze(1)], dim=1)

            loss.backward()
            optimizer.step()

        fold_train_losses.append(total_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            pred_test = model(X_test_tensor)
            test_loss = loss_cont(pred_test, Y_test_tensor).item()
            fold_test_losses.append(test_loss)
        
            scheduler.step(test_loss)
            print(f"Epoch {epoch+1}, current lr: {optimizer.param_groups[0]['lr']:.6f}")

            
            if epoch == num_epochs - 1:
                b, t, f = pred_test.shape
                pred_test_flat = pred_test.cpu().numpy().reshape(b*t, f)
                Y_test_flat = Y_test_tensor.cpu().numpy().reshape(b*t, f)

                # inverse transform
                pred_test_inv = scaler_Y.inverse_transform(pred_test_flat)
                Y_test_inv = scaler_Y.inverse_transform(Y_test_flat)

                # R² po cechach
                r2s = [r2_score(Y_test_inv[:, i], pred_test_inv[:, i]) for i in range(f)]
                avg_r2_per_output = np.mean(r2s)
                print("Avg R2 per output:", r2s)
                print("Mean R2:", avg_r2_per_output)

                # === DIAGNOSTYKA PER-FEATURE ===
                from sklearn.metrics import mean_squared_error
                print("\n--- DIAGNOSTYKA ---")

                # przygotuj dane surowe z treningu (przed skalowaniem!)
                # jeśli chcesz pełną dokładność, przechowuj Y_train_raw = Y_train (niezeskalowane) tuż po wczytaniu
                # tu tymczasowo przyjmujemy, że masz Y_train nieprzeskalowane dostępne jako oryginalne dane:
                flat_y_train = np.vstack([y for y in Y_train_raw])  # jeśli Y_train jest już scaled, zmień to na surowe Y_train_raw

                vars_train = flat_y_train.var(axis=0)
                means_train = flat_y_train.mean(axis=0)
                mins = flat_y_train.min(axis=0)
                maxs = flat_y_train.max(axis=0)

                print("Feature | var | mean | min | max")
                for i in range(f):
                    print(i, round(vars_train[i],6), round(means_train[i],6), round(mins[i],6), round(maxs[i],6))

                # baseline R² (predykcja średnią z train)
                mean_pred = np.tile(means_train.reshape(1,-1), (Y_test_inv.shape[0],1))
                baseline_r2 = [r2_score(Y_test_inv[:,i], mean_pred[:,i]) for i in range(f)]
                print("\nBaseline R2 per feature:", baseline_r2)

                # model R² + MSE + min/max pred
                for i in range(f):
                    r2 = r2_score(Y_test_inv[:,i], pred_test_inv[:,i])
                    mse = mean_squared_error(Y_test_inv[:,i], pred_test_inv[:,i])
                    print(f"feat {i}: R2={r2:.4f}, MSE={mse:.6f}, pred_min={pred_test_inv[:,i].min():.4f}, pred_max={pred_test_inv[:,i].max():.4f}")
                print("--- KONIEC DIAGNOSTYKI ---\n")

                num_features = Y_test_inv.shape[1]
                fig, axes = plt.subplots(nrows=(num_features + 2)//3, ncols=3, figsize=(15, 5*((num_features + 2)//3)))
                axes = axes.flatten()
                titles = ["Race progress","Lap progress","Fuel level","Tyre wear FL","Tyre wear FR","Tyre wear RL","Tyre wear RR",
                            "Tyre temp FL","Tyre temp FR","Tyre temp RL","Tyre temp RR","Track wetness"]
                for i in range(num_features):
                    ax = axes[i]
                    ax.scatter(Y_test_inv[:, i], pred_test_inv[:, i], s=10, alpha=0.6)
                    ax.plot([Y_test_inv[:, i].min(), Y_test_inv[:, i].max()],
                            [Y_test_inv[:, i].min(), Y_test_inv[:, i].max()],
                            'r--', linewidth=1.5, label='ideal line')
                    ax.set_title(f'{titles[i]} | R²={r2s[i]:.3f}')
                    ax.set_xlabel('True')
                    ax.set_ylabel('Predicted')
                    ax.legend()
                    ax.grid(True)

                for j in range(i+1, len(axes)):
                    fig.delaxes(axes[j])

                plt.tight_layout()
                plt.show()

                num_features = Y_test_inv.shape[1]
                titles = ["Race progress","Lap progress","Fuel level",
                        "Tyre wear FL","Tyre wear FR","Tyre wear RL","Tyre wear RR",
                        "Tyre temp FL","Tyre temp FR","Tyre temp RL","Tyre temp RR",
                        "Track wetness"]

                cols = 3
                rows = int(np.ceil(num_features / cols))
                fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 3))
                axes = axes.flatten()

                for i in range(num_features):
                    ax = axes[i]
                    ax.plot(Y_test_inv[:, i], label='True', color='blue', linewidth=1)
                    ax.plot(pred_test_inv[:, i], label='Pred', color='orange', linestyle='--', linewidth=1)
                    ax.set_title(f'{titles[i]} | R²={r2s[i]:.3f}')
                    ax.set_xlabel('Sample index')
                    ax.set_ylabel('Value')
                    ax.legend()
                    ax.grid(True)

                # usuń puste osie, jeśli niepełny ostatni rząd
                for j in range(i+1, len(axes)):
                    fig.delaxes(axes[j])

                plt.tight_layout()
                plt.show()
                                                
    plt.figure()
    plt.plot(fold_train_losses, label='Train')
    plt.plot(fold_test_losses, label='Test')
    plt.title(f'Fold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

del model, X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor
torch.cuda.empty_cache()

    # if i == 1:
    #     break

                # opcjonalnie: wróć do kształtu 3D, jeśli chcesz generować sekwencje
                # pred_test_inv = pred_test_inv.reshape(b, t, f)
                # Y_test_inv    = Y_test_inv.reshape(b, t, f)


        
    # all_fold_train.append(fold_train_losses)
    # all_fold_test.append(fold_test_losses)
    # # all_fold_r2.append(fold_r2_scores)

    # # Compute R^2 score for the test set
    # r2 = r2_score(Y_test_tensor.cpu().numpy(), pred_test.cpu().numpy())
    # all_fold_r2_test.append(r2)
