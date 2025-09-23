import json
from telemetry.LMU_plugin import collect_telemetry
from envs.filtr_json_from_race import save_to_db, load_from_db, delete_from_db
# from envs.racing_env import RacingEnv
# from ai.BCmodel import BCModel, load_data, save_model
import torch
# from ai.state_predictor import RaceRegressor
import numpy as np
import time
import os
from pathlib import Path


if __name__ == "__main__":
    # usage_multiplier = 1.0  # Ustawienia wyścigu
    # collect_telemetry(usage_multiplier)
    

    data_dir = Path(r"E:/pracadyp/Race-optimization-reinforcement-learning/data/raw_races")

    for telem_path in data_dir.glob("telemetry_data*.json"):
        # Tworzymy odpowiadającą nazwę scoring
        scoring_path = data_dir / telem_path.name.replace("telemetry", "scoring")
        if scoring_path.exists():
            save_to_db(str(telem_path), str(scoring_path))


 
    
    load_from_db()

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

    # print("✅ Plik został zaktualizowany (multiplier = 2.0)")