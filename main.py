from telemetry.LMU_plugin import collect_telemetry
from envs.filtr_json_from_race import filtr_json_files
# from envs.racing_env import RacingEnv
# from ai.BCmodel import BCModel, load_data, save_model
import torch
# from ai.state_predictor import RaceRegressor
import numpy as np
import time
import os


if __name__ == "__main__":
    usage_multiplier = 3.0  # Ustawienia wyścigu
    collect_telemetry(usage_multiplier)

    # filtr_json_files()
    