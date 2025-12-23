import numpy as np
import socket
import threading
import json
from queue import Queue, Empty
import time
from torch.distributions import Categorical
import torch
from datetime import datetime
import os


def run_rl_agent(client, model, scaler_X_min_max, scaler_X_robust, usage_multiplier=3.0, save_dir="telemetry_logs"):
    print("Start wątku RL - tryb: Scoring -> Next Telem")

    # Zmienna-magazyn: tu trzymamy Scoring, który czeka na swoją parę (Telemetrię)
    pending_scoring = None
    
    # Dla logowania: magazyn dla każdego scoring (niezależnie od sektora)
    last_scoring = None
    
    # Do wykrywania zmiany sektora
    prev_sector = -1
    
    # Listy do zbierania scoring i telemetry OSOBNO
    scoring_log = []
    telemetry_log = []
    record_counter = 0
    scoring_counter = 0  # Licznik wszystkich scoringów (do filtrowania co 2)
    
    # Utwórz katalog na logi
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scoring_file = os.path.join(save_dir, f"race_scoring_{timestamp}.json")
    telemetry_file = os.path.join(save_dir, f"race_telemetry_{timestamp}.json")
    print(f"📊 Zapisywanie danych do:")
    print(f"   Scoring:    {scoring_file}")
    print(f"   Telemetry:  {telemetry_file}")

    try:
        while client.running:
            try:
                # Czekamy na dane (nie blokujemy procesora na 100%, ale reagujemy natychmiast)
                data = client.queue.get(timeout=1.0)
            except Empty:
                continue

            msg_type = data.get("Type")

            # --- 1. Przyszło SCORING ---
            if msg_type == "ScoringInfoV01":
            
                # Pobieramy dane gracza
                vehicles = data.get("mVehicles", [])
                # Szybkie szukanie gracza
                player = next((v for v in vehicles if v.get("mIsPlayer")), None)

                if not player:
                    continue
                
                # ========================================
                # LOGOWANIE: Zapisz CO DRUGI scoring
                # ========================================
                scoring_counter += 1
                if scoring_counter % 2 == 0:  # Co drugi scoring
                    last_scoring = data.copy()
                    last_scoring["mVehicles"] = [player]  # Tylko gracz, nie wszystkie pojazdy
                    record_counter += 1
                    scoring_record = {
                        "record_id": record_counter,
                        "timestamp": datetime.now().isoformat(),
                        "data": last_scoring
                    }
                    scoring_log.append(scoring_record)
                # ========================================
                
                curr_sector = player["mSector"]
                
                # Debug: Pokaż zmianę sektora
                if curr_sector != prev_sector:
                    print(f"🏁 Sektor: {prev_sector} → {curr_sector} (Okr: {player['mTotalLaps']})")
            
                # WARUNEK WYZWOLENIA:
                # Właśnie wjechaliśmy w sektor 2 (a wcześniej byliśmy w innym, np. 1)
                # I NIE mamy już oczekującego scoringu (żeby nie nadpisać go dwa razy w tej samej sekundzie)
                if curr_sector == 0 and prev_sector == 2 and pending_scoring is None:
                    print(f"\n{'='*60}")
                    print(f"⚡ TRIGGER! Sektor 0 po 2 - Okrążenie {player['mTotalLaps']}")
                    print(f"{'='*60}")
                    
                    # Przygotowujemy dane scoringu pod extrakcję
                    data["mVehicles"] = [player]
                    pending_scoring = data

                # Aktualizujemy historię sektora
                prev_sector = curr_sector

            # --- 2. Przyszło TELEM ---
            elif msg_type == "TelemInfoV01":
                
                # ========================================
                # LOGOWANIE: Zapisz TYLKO PIERWSZĄ telemetry po każdym scoringu
                # ========================================
                if last_scoring is not None:
                    telemetry_record = {
                        "record_id": record_counter,  # Ten sam co scoring
                        "timestamp": datetime.now().isoformat(),
                        "data": data
                    }
                    telemetry_log.append(telemetry_record)
                    last_scoring = None  # Reset - następne telemetrie do nowego scoringu
                    
                    # Auto-zapis co 20 par
                    if record_counter % 20 == 0:
                        with open(scoring_file, 'w') as f:
                            json.dump(scoring_log, f, indent=2)
                        with open(telemetry_file, 'w') as f:
                            json.dump(telemetry_log, f, indent=2)
                    
                    if record_counter % 100 == 0:
                        print(f"💾 [{record_counter}] Auto-zapis: {len(scoring_log)} par")
                # ========================================
                
                # Czy mamy oczekujący Scoring? (Czy "zapadka" jest ustawiona?)
                if pending_scoring is not None:
                    # TO JEST TEN MOMENT - Pierwsza telemetria po scoringu
                    
                    try:
                        # Dodajemy multiplier do telemetrii
                        data["multiplier"] = usage_multiplier
                        
                        # 1. Łączymy zapamiętany Scoring z bieżącą Telemetrią
                        raw_state = extract_state(data, pending_scoring)
                        
                        # 2. Skalowanie
                        input_vector = preprocess_data(np.array(raw_state), scaler_X_min_max, scaler_X_robust)
                        
                    
                        tensor_in = torch.FloatTensor(input_vector).unsqueeze(0)
                        
                        with torch.no_grad():
                            print("Obliczam akcję modelu...")
                            # Zakładam, że model zwraca logity lub akcje
                            prediction = select_action_deterministic(model, input_vector)
                            
                            wheels = ["Soft", "Medium", "Hard", "Wet"]
                        
                            print("Na podstawie stanu:")
                            print("Ilość paliwa:" , raw_state[0])
                            print("Postęp wyścigu:", raw_state[1])
                            print("Zużycie opon LF:", raw_state[2])
                            print("Zużycie opon RF:", raw_state[3])
                            print("Zużycie opon LR:", raw_state[4])
                            print("Zużycie opon RR:", raw_state[5])
                            print("Wilgotność toru:", raw_state[6])
                            print("Natężenie deszczu:", raw_state[7])
                            print("Uszkodzenia nadwozia:", raw_state[8:16])
                            print("Liczba okrążeń:", raw_state[16])
                            print("Liczba pit-stopów:", raw_state[17])
                            print("Typ opon:", wheels[int(raw_state[18])])
                            print("Mnożnik zużycia:", raw_state[19])
                            print("Temperatura opon LF:", raw_state[20])
                            print("Temperatura opon RF:", raw_state[21])
                            print("Temperatura opon LR:", raw_state[22])
                            print("Temperatura opon RR:", raw_state[23])
                            print("Temperatura otoczenia:", raw_state[24])
                            print("Temperatura toru:", raw_state[25])
                            print("Przewidywany czas zakończenia wyścigu:", raw_state[26])

                            if prediction[0] == 1:
                                print("Decyzja: Wjazd na pit-stop")

                            
                            print(f"Action: {action_to_string(prediction)}")
                        

                    except Exception as e:
                        print(f"Błąd w obliczeniach RL: {e}")
                    
                
                    pending_scoring = None   

    except KeyboardInterrupt:
        print("\n⚠️ Przerwano przez użytkownika (Ctrl+C)")
    
    finally:
        # ========================================
        # ZAPIS KOŃCOWY - wykonuje się ZAWSZE
        # ========================================
        print(f"\n{'='*60}")
        print(f"🏁 Koniec sesji - zapisuję dane końcowe...")
        if scoring_log or telemetry_log:
            with open(scoring_file, 'w') as f:
                json.dump(scoring_log, f, indent=2)
            with open(telemetry_file, 'w') as f:
                json.dump(telemetry_log, f, indent=2)
            print(f"✅ Zapisano:")
            print(f"   Scoring:    {len(scoring_log)} rekordów -> {scoring_file}")
            print(f"   Telemetry:  {len(telemetry_log)} rekordów -> {telemetry_file}")
        else:
            print("⚠️ Brak rekordów do zapisania")
        print(f"{'='*60}\n")


def preprocess_data(raw_vector_x, scaler_X_min_max, scaler_X_robust):
    """
    Skaluje pojedynczy wektor (37,), stosując scaler tylko do 
    części ciągłej (0-19) i zostawiając kategorialną (20-36).
    """
    no_scaler_x = slice(0, 8)  # no scaler for X
    min_max_scaler_x = slice(8, 20)  # min-max scaler for X
    robust_scaler_x = slice(20, 28)  # robust scaler for X
    #POTEM JAK ZMIENIE NA NORM ENDET
    # no_scaler_x = slice(0, 9)  # no scaler for X
    # min_max_scaler_x = slice(9, 21)  # min-max scaler for X
    # robust_scaler_x = slice(21, 28)  # robust scaler for X
    
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
    vehicles = scoring_raw.get("mVehicles", [])
    
    player_vehicle = None
    for v in vehicles:
        if v.get("mIsPlayer") == True:
            player_vehicle = v
            break

    # ✅ DODAJ SPRAWDZENIE
    if not player_vehicle:
        raise ValueError("❌ Nie znaleziono gracza w danych Scoring!")

    subset_scoring_vehicle = {k: player_vehicle.get(k) for k in wanted_keys}
    
    wanted_keys_telem = ["mFuel", "mFuelCapacity","mWheel","mDentSeverity","mFrontTireCompoundIndex","mCurrentSector","mLapNumber","mLastImpactET","mLastImpactMagnitude","multiplier","is_repairing"]
    subset_telem = {k: telem_raw.get(k) for k in wanted_keys_telem}

    filtered_data_scoring = {**subset_scoring_vehicle, **subset_weather}
    filtered_data_telemetry = subset_telem

    return filtered_data_telemetry, filtered_data_scoring
def extract_state(telem_file_raw, scoring_file_raw):
        filtered_data_telemetry, filtered_data_scoring = filtr_data(telem_file_raw,scoring_file_raw)
        data_state = []
        
        scoring = filtered_data_scoring
        telemetry = filtered_data_telemetry
        
        
        data_state_rl = [
            
            telemetry["mFuel"]/telemetry["mFuelCapacity"],
            scoring["mCurrentET"]/scoring["mEndET"],
            telemetry['mWheel'][0]['mWear'],  
            telemetry["mWheel"][1]["mWear"],
            telemetry["mWheel"][2]["mWear"],
            telemetry["mWheel"][3]["mWear"],
            scoring["mAvgPathWetness"],
            scoring["mRaining"],
            # round(scoring["mEndET"],5)/7200.0, potem jak zmienie na norm

            
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



        return data_state_rl


def select_action_deterministic(model, state):
    """Wersja dla rzeczywistych wyścigów - wybiera najlepszą akcję"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_list, _ = model(state_tensor)

    # DEBUG: Pokaż surowe logity
    print(f"\n🔍 DEBUG - Surowe logity modelu:")
    print(f"   Pit-stop logits: {logits_list[0].cpu().numpy()}")
    print(f"   Tire logits: {logits_list[1].cpu().numpy()}")
    print(f"   Repair logits: {logits_list[2].cpu().numpy()}")
    print(f"   Fuel logits: {logits_list[3].cpu().numpy()}")

    actions = []
    for logits in logits_list:
        # Najbardziej prawdopodobna akcja
        action = logits.squeeze(0).argmax()
        actions.append(action.item())

    return actions

def action_to_string(actions):
    """Zwraca zwięzły string z akcją"""
    if actions[0] == 0:
        return "Brak pit-stopu"
    pit = "Zjedź na pit-stop" if actions[0] == 1 else "Nie zjeżdżaj"
    tire_names = ["Bez zmiany", "Miękkie", "Średnie", "Twarde", "Deszczowe"]
    repair = "Naprawa" if actions[2] == 1 else "Brak naprawy"
    fuel_pct = actions[3] * 20
    
    return f"{pit} | {tire_names[actions[1]]} | {repair} | Paliwo:{fuel_pct}%"

