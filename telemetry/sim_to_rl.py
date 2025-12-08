import numpy as np
import socket
import threading
import json
from queue import Queue, Empty
import time
from pandas import Categorical
import torch


def run_rl_agent(client, model, scaler_X_min_max, scaler_X_robust, usage_multiplier=1.0):
    print("Start wątku RL - tryb: Scoring -> Next Telem")

    # Zmienna-magazyn: tu trzymamy Scoring, który czeka na swoją parę (Telemetrię)
    pending_scoring = None
    
    # Do wykrywania zmiany sektora
    prev_sector = -1

    while client.running:
        try:
            # Czekamy na dane (nie blokujemy procesora na 100%, ale reagujemy natychmiast)
            data = client.queue.get(timeout=1.0)
        except Empty:
            continue

        msg_type = data.get("Type")

        # --- 1. Przyszło SCORING (Sprawdzamy czy to moment decyzji) ---
        if msg_type == "ScoringInfoV01":
            
            # Pobieramy dane gracza
            vehicles = data.get("mVehicles", [])
            # Szybkie szukanie gracza
            player = next((v for v in vehicles if v.get("mIsPlayer")), None)

            if not player:
                continue
            
            curr_sector = player["mSector"]

            # WARUNEK WYZWOLENIA:
            # Właśnie wjechaliśmy w sektor 2 (a wcześniej byliśmy w innym, np. 1)
            # I NIE mamy już oczekującego scoringu (żeby nie nadpisać go dwa razy w tej samej sekundzie)
            if curr_sector == 2 and prev_sector != 2 and pending_scoring is None:
                print(f"TRIGGER: Wjazd w Sektor 2 (Lap {player['mTotalLaps']}). Czekam na pierwszą telemetrię...")
                
                # Przygotowujemy dane scoringu pod extrakcję
                # (podmieniamy listę pojazdów na samego gracza, żeby extract_state zadziałał)
                data["mVehicles"] = [player]
                
                # ZATRZASK: Zapisujemy scoring i czekamy na następny pakiet Telem
                pending_scoring = data

            # Aktualizujemy historię sektora
            prev_sector = curr_sector

        # --- 2. Przyszło TELEM (Sprawdzamy czy mamy na co odpowiadać) ---
        elif msg_type == "TelemInfoV01":
            
            # Czy mamy oczekujący Scoring? (Czy "zapadka" jest ustawiona?)
            if pending_scoring is not None:
                # TO JEST TEN MOMENT - Pierwsza telemetria po scoringu
                
                try:
                    # Dodajemy multiplier do telemetrii
                    data["multiplier"] = usage_multiplier
                    
                    # 1. Łączymy zapamiętany Scoring z bieżącą Telemetrią
                    # extract_state zwraca listę [wartość, wartość, ...]
                    raw_state = extract_state(data, pending_scoring)
                    
                    # 2. Skalowanie
                    # preprocess_data zwraca spłaszczony numpy array
                    input_vector = preprocess_data(np.array(raw_state), scaler_X_min_max, scaler_X_robust)
                    
                    # 3. Predykcja modelu
                    # Zamiana na Tensor PyTorch [1, wymiar]
                    tensor_in = torch.FloatTensor(input_vector).unsqueeze(0)
                    
                    with torch.no_grad():
                        # Zakładam, że model zwraca logity lub akcje
                        prediction = select_action(model, input_vector)
                        
                        # Tutaj logika wyciągania akcji (zależnie od tego co zwraca Twój model)
                        # np. dla MultiDiscrete często robi się argmax na wynikach
                        # action = ... (zostawiam printa, bo zależy od modelu)
                        
                        print(f"Model Action: {prediction}")
                        # send_to_game(action) 

                except Exception as e:
                    print(f"Błąd w obliczeniach RL: {e}")
                
                # WAŻNE: Resetujemy zatrzask!
                # Dzięki temu nie wyślemy 10 decyzji pod rząd, tylko jedną na okrążenie.
                pending_scoring = None   

            


def preprocess_data(raw_vector_x, scaler_X_min_max, scaler_X_robust):
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


def filtr_data(telem_raw, scoring_raw):


    wanted_weather_keys = ["mRaining","mAmbientTemp","mTrackTemp","mEndET", "mCurrentET","mAvgPathWetness"]
    subset_weather = {k: scoring_raw.get(k) for k in wanted_weather_keys}
    subset_weather["mTotalLapDistance"] = scoring_raw["mLapDist"]

    wanted_keys = ["mLastLapTime","mBestLapTime","mCurrLapTime","mNumPitstops","mNumPenalties","mInPits","mFinishStatus","mLapDist","mSector","mTotalLaps"]
        
    subset_scoring_vehicle = {k: vehicle[0].get(k) for k in wanted_keys}


    wanted_keys_telem = ["mFuel", "mFuelCapacity","mWheel","mDentSeverity","mFrontTireCompoundIndex","mCurrentSector","mLapNumber","mLastImpactET","mLastImpactMagnitude","multiplier","is_repairing"]
    vehicle = scoring_raw.get("mVehicles")

    subset_telem = {k: telem_raw.get(k) for k in wanted_keys_telem}

    filtered_data_scoring = {**subset_scoring_vehicle, **subset_weather}
    filtered_data_telemetry = subset_telem

    return filtered_data_telemetry, filtered_data_scoring
def extract_state(telem_file_raw, scoring_file_raw):
        filtered_data_telemetry, filtered_data_scoring = filtr_data(telem_file_raw,scoring_file_raw)
        data_state = []
        
        scoring = filtered_data_scoring
        telemetry = filtered_data_telemetry
        
        
        data_state = [
            
            telemetry["mFuel"]/telemetry["mFuelCapacity"],
            scoring["mCurrentET"]/scoring["mEndET"],
            telemetry['mWheel'][0]['mWear'],  
            telemetry["mWheel"][1]["mWear"],
            telemetry["mWheel"][2]["mWear"],
            telemetry["mWheel"][3]["mWear"],
            scoring["mAvgPathWetness"],
            scoring["mRaining"],

            
            #MIN-MAX SCALER
            
            telemetry["mDentSeverity"][0],  # Not defined which part of the car this refers to each index
            telemetry["mDentSeverity"][1],
            telemetry["mDentSeverity"][2], 
            telemetry["mDentSeverity"][3],
            telemetry["mDentSeverity"][4],
            telemetry["mDentSeverity"][5],
            telemetry["mDentSeverity"][6], 
            telemetry["mDentSeverity"][7],
            scoring["mTotalLaps"],
            scoring["mNumPitstops"],
            telemetry["mFrontTireCompoundIndex"],
            telemetry["multiplier"],
            #ROUBST SCALER
            sum(telemetry["mWheel"][0]["mTemperature"])/len(telemetry["mWheel"][0]["mTemperature"]),
            sum(telemetry["mWheel"][1]["mTemperature"])/len(telemetry["mWheel"][1]["mTemperature"]),
            sum(telemetry["mWheel"][2]["mTemperature"])/len(telemetry["mWheel"][2]["mTemperature"]),
            sum(telemetry["mWheel"][3]["mTemperature"])/len(telemetry["mWheel"][3]["mTemperature"]),
            scoring["mAmbientTemp"],
            scoring["mTrackTemp"],
            round(scoring["mEndET"],5),
        ]





            
            
    
          
           

       
        # with open("data/state_data.json", "w") as file:
        #     json.dump(data_state, file, indent=2)

        return data_state


def select_action(model, state):
    """
    Poprawnie wybiera akcję, używając obiektu Categorical,
    aby zapewnić spójność z pętlą 'ppo_update'.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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

