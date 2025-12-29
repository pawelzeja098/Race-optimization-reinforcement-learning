import json
from telemetry.LMU_plugin import collect_telemetry
from envs.filtr_json_from_race import save_to_db, load_from_db, delete_from_db
from envs.filtr_json_test_data import save_to_db_tests 
# from envs.racing_env import RacingEnv
# from ai.BCmodel import BCModel, load_data, save_model
import torch
# from ai.state_predictor import RaceRegressor
import numpy as np
import time
import os
from pathlib import Path
import socket
import threading
import json
from queue import Queue, Empty
import time
from telemetry.sim_to_rl import run_rl_agent
from telemetry.LMU_plugin import TelemetryClient
import joblib
from ai.RLmodel import ActorCritic
from gymnasium import spaces

if __name__ == "__main__":
    # --------- COLLECT TELEMETRY DATA ---------
    # usage_multiplier = 3.0  # Ustawienia wyścigu
    # collect_telemetry(usage_multiplier)

    #---------- USE RL AGENT TO MAKE DECISIONS ----------
    # action_space = spaces.MultiDiscrete([
    #                                     2, # Pit stop or not
    #                                     5, # Tire change (0-4) No, soft, medium, hard, wet
    #                                     2, # Repair or not (0-1)
    #                                     6, # Fuel * 0.2 (0-20)
    #                                     ])
    # state_dim = 27

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Używane urządzenie: {device}")
    
    # # ✅ 2. Utwórz model i od razu przenieś go na device
    # model = ActorCritic(state_dim, action_space).to(device)
   
    # # model = ActorCritic(state_dim, action_space) 

    
    # path = "models/RL_agent12_final.pth"
    # checkpoint = torch.load(path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # try:
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     print(f"Pomyślnie wczytano wagi z epoki: {checkpoint.get('epoch', '?')}")
    # except RuntimeError as e:
    #     print(f"Błąd kształtu wag! Czy zmieniłeś architekturę modelu? {e}")

 
    # model.eval()

    # scaler_minmax_X = joblib.load("models/scalerX_min_max_RL12.pkl")
    # scaler_robust_X = joblib.load("models/scalerX_robust_RL12.pkl")

    # client = TelemetryClient()
    # client.start()  # To uruchamia wątek odbierania (Wątek 1)

    # # To uruchamia wątek przetwarzania RL (Wątek 2)
    # rl_thread = threading.Thread(target=run_rl_agent, args=(client, model, scaler_minmax_X, scaler_robust_X), daemon=True)
    # rl_thread.start()

    # print("System działa. Główny wątek jest wolny.")

    # # --- Wątek Główny (Main) ---
    # # Tutaj możesz robić co chcesz: obsługiwać GUI, czekać na klawisz wyjścia itp.
    # try:
    #     while True:
    #         time.sleep(1) # Tylko po to, żeby główny wątek nie zużywał 100% CPU
    # except KeyboardInterrupt:
    #     print("Zamykanie...")
    #     client.stop()
    #     # Wątki daemon same zginą po zamknięciu głównego
    

    #--------- FILTER JSON FILES FROM RAW TRAIN RACES ---------

    # data_dir = Path(r"E:/pracadyp/Race-optimization-reinforcement-learning/data/raw_races")

    # for telem_path in data_dir.glob("telemetry_data*.json"):
    #     # Tworzymy odpowiadającą nazwę scoring
    #     scoring_path = data_dir / telem_path.name.replace("telemetry", "scoring")
    #     if scoring_path.exists():
    #         save_to_db(str(telem_path), str(scoring_path))

    #--------- FILTER JSON FILES FROM RAW TEST RACES ---------
    data_dir = Path(r"E:/pracadyp/Race-optimization-reinforcement-learning/telemetry_logs")


    for telem_path in data_dir.glob("race_telemetry*.json"):
        # Tworzymy odpowiadającą nazwę scoring
        scoring_path = data_dir / telem_path.name.replace("telemetry", "scoring")
        if scoring_path.exists():
            save_to_db_tests(str(telem_path), str(scoring_path))



 
    
    # load_from_db()

    # filtr_json_files()
    
    # telemetry = "data/raw_races/telemetry_data.json"
    # with open(telemetry, "r", encoding="utf-8") as f:
    #     data = json.load(f)

    # # jeśli plik zawiera listę obiektów:
    # if isinstance(data, list):
    #     for obj in data:
    #         if "multiplier" in obj:
    #             obj["multiplier"] = 2.0
    #         if "usage multiplier" in obj:
    #             obj["usage multiplier"] = 2.0

    # # jeśli plik zawiera pojedynczy obiekt:
    # elif isinstance(data, dict):
    #     if "multiplier" in data:
    #         data["multiplier"] = 2.0
    #     if "usage multiplier" in data:
    #         data["usage multiplier"] = 2.0

    # # zapisujemy z powrotem w to samo miejsce
    # with open(telemetry, "w", encoding="utf-8") as f:
    #     json.dump(data, f, indent=2, ensure_ascii=False)

    # print(" Plik został zaktualizowany (multiplier = 2.0)")