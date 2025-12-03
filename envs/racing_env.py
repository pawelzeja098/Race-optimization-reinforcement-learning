import os
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json

matplotlib.use('TkAgg')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gymnasium as gym
from gymnasium import spaces
from envs.weather_generator import generate_weather_conditions
from envs.impact_generator import random_impact_magnitude, generate_dent_severity
from ai.LSTMmodel import LSTMStatePredictor, generate_predictions
import torch
from collections import deque
from config import X_SHAPE, Y_SHAPE, CONT_LENGTH, CAT_LENGTH

data_race_scoring = "data/scoring_data.json"
data_race_telemetry = "data/telemetry_data.json"
probabilities = np.load('E:/pracadyp/Race-optimization-reinforcement-learning/data/probabilities_impact/probabilities.npy')
bin_edges = np.load('E:/pracadyp/Race-optimization-reinforcement-learning/data/probabilities_impact/bin_edges.npy')

class RacingEnv(gym.Env):
    def __init__(self):
        super(RacingEnv,self).__init__()
        self.history = []
        self.lap_dist = 0.0
        self.race_complete_perc = 0.0
        self.fuel_tank_capacity = -1.0
        self.wheel1_wear = -1.0
        self.wheel2_wear = -1.0
        self.wheel3_wear = -1.0
        self.wheel4_wear = -1.0
        self.wheel1_delta = 0.0
        self.wheel2_delta = 0.0
        self.wheel3_delta = 0.0
        self.wheel4_delta = 0.0
        self.wheel1_temp = -1.0
        self.wheel2_temp = -1.0
        self.wheel3_temp = -1.0
        self.wheel4_temp = -1.0
        self.path_wetness = 0.0
        self.last_impact_et = 0.0
        self.last_impact_magnitude = 0.0
        self.num_penalties = 0.0
        self.raining = 0.0
        self.ambient_temp = 0.0
        self.track_temp = 0.0
        self.end_et = 0.0
        self.dent_severity = [0.0]*8
        self.has_last_lap = 0.0
        self.finish_status = 0.0
        # self.total_laps = 0.0
        self.sector = 0.0
        self.num_pit_stops = 0.0
        self.in_pits = 0.0
        self.tire_compound_index = 0.0
        self.changed_tires_flag = 0.0
        self.refueled_flag = 0.0
        self.usage_multiplier = 1.0
        self.prev_et = 0.0
        self.curr_step = 0
        self.total_steps = 5007
        self.impact_flag = 0.0
        self.pitted = False
        self.delta = 0.0
        self.num_race = 0
        self.is_repairing = 0.0
        # self.weather_conditions = generate_weather_conditions(self.total_steps)
        self.pit_stage = 0
        self.last_lap_step = 0
        self.laps = 0
        self.checked_pit = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state = None
        # self.scaler_X = joblib.load("models/scaler2_X.pkl")
        self.scaler_minmax_X = joblib.load("models/scalerX_min_max.pkl")
        self.scaler_robust_X = joblib.load("models/scalerX_robust.pkl")
        self.scaler_minmax_Y = joblib.load("models/scalerY_min_max.pkl")
        self.scaler_robust_Y = joblib.load("models/scalerY_robust.pkl")
        # self.scaler_Y = joblib.load("models/scaler2_Y.pkl")
        self.h_c = None
        self.lap_checked = False
        self.target_fuel = 0.0

        self.LSTM_model = LSTMStatePredictor(input_size=X_SHAPE, hidden_size=256, output_size=Y_SHAPE, num_layers=1).to(device)
        self.LSTM_model.load_state_dict(torch.load("models/lstmdeltaT_model.pth", map_location=device))
        self.LSTM_model.eval()
        # self.curr_window = deque(maxlen=30)

        # self.impact_magnitude_history = []
        # self.impact_flag_history = []
        # self.dent_severity_history = []
        # self.dent_severity_change = [0.0]*8
        # for i in range(self.total_steps):

        #     impact_magnitude = random_impact_magnitude(probabilities=probabilities, bin_edges=bin_edges)
        #     if impact_magnitude > 0.0:
        #         impact_flag = 1.0
        #         self.dent_severity_change = generate_dent_severity(impact_magnitude)
        #     else:
        #         impact_flag = 0.0
        #         self.dent_severity_change = [0.0]*8
                    
            
        #     self.impact_magnitude_history.append(impact_magnitude)
        #     self.impact_flag_history.append(impact_flag)
        #     self.dent_severity_history.append(self.dent_severity_change)
               

        #New obs space including only data aviable directly from simulator. 
        self.observation_space = gym.spaces.Box(
    low=np.array([
        0.0,   # Tank capacity
        0.0,   # Path wetness
        0.0,   # Wheel wear
        0.0,   # Wheel wear
        0.0,   # Wheel wear
        0.0,   # Wheel wear
        0.0,   #step ratio
        0.0,   # Raining

        
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # dent severities
        0.0,   # Total laps
        0.0,   # Num pitstops
        0.0,   # tire compound index
        1.0,   # multiplier

        100.0,   # Wheel temperature
        100.0,   # Wheel temperature
        100.0,   # Wheel temperature
        100.0,   # Wheel temperature   
        3.0,   # Ambient temp
        8.0,   # Track temp
        0.0,   # End ET 

    ], dtype=np.float32),
    high=np.array([
        1.1,     # Tank capacity
        1.0,    # Path wetness
        1.0,     # Wheel wear
        1.0,     # Wheel wear
        1.0,     # Wheel wear
        1.0,     # Wheel wear
        1.0,     #step ratio
        1.0,   # Raining
        
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,  # dent severities
        400.0,   # Total laps
        100.0,   # Num pitstops
        3.0,    # tire compound index
        3.0,   # multiplier
    
        600.0,   # Wheel temperature
        600.0,   # Wheel temperature
        600.0,   # Wheel temperature
        600.0,   # Wheel temperature
        45.0,   # Ambient temp
        60.0,   # Track temp
        86500.0,   # End ET
    ], dtype=np.float32),
    dtype=np.float32
)
    
        

        self.action_space = spaces.MultiDiscrete([
                                                2, # Pit stop or not
                                                # 2, # Confirm pit stop or not
                                                5, # Tire change (0-4) No, soft, medium, hard, wet
                                                2, # Repair or not (0-1)
                                                6, # Fuel * 0.2 (0-20)
                                                ])
        
   


    def reset(self,end_et = 1932.0,total_steps=5007,usage_multiplier=1.0,no_rain=False):
        self.total_steps = total_steps
        self.weather_conditions = generate_weather_conditions(self.total_steps,no_rain=no_rain)
        self.impact_magnitude_history = []
        self.impact_flag_history = []
        self.dent_severity_history = []
        self.dent_severity_change = [0.0]*8
        
        for i in range(self.total_steps):

            impact_magnitude = random_impact_magnitude(probabilities=probabilities, bin_edges=bin_edges)
            if impact_magnitude > 0.0:
                impact_flag = 1.0
                self.dent_severity_change = generate_dent_severity(impact_magnitude)
            else:
                impact_flag = 0.0
                self.dent_severity_change = [0.0]*8
                    
            
            self.impact_magnitude_history.append(impact_magnitude)
            self.impact_flag_history.append(impact_flag)
            self.dent_severity_history.append(self.dent_severity_change)
        
        self.end_et = end_et
        self.target_fuel = 0.0
        # self.total_steps = total_steps
        self.usage_multiplier = usage_multiplier


        self.laps = 0
        
        self.lap_dist = 0.0005
        # self.race_complete_perc = 0.0
        self.fuel_tank_capacity = 1.0
        self.wheel1_wear = 1.0
        self.wheel2_wear = 1.0
        self.wheel3_wear = 1.0
        self.wheel4_wear = 1.0
        self.wheel1_delta = 0.0
        self.wheel2_delta = 0.0
        self.wheel3_delta = 0.0
        self.wheel4_delta = 0.0
        self.wheel1_temp = 310.0
        self.wheel2_temp = 310.0
        self.wheel3_temp = 312.0
        self.wheel4_temp = 312.0
        self.currently_in_pit = False
        
        self.last_impact_et = 0.0
        self.last_impact_magnitude = 0.0
        self.num_penalties = 0.0
        self.raining = 0.0
        self.ambient_temp = 0.0
        self.track_temp = 0.0
        # self.end_et = 0.0
        # self.dent_severity = [0.0]*8
        # self.has_last_lap = 0.0
        self.finish_status = 0.0
        self.laps = 0.0
        self.sector = 1.0
        self.num_pit_stops = 0.0
        self.in_pits = 0.0
        self.tire_compound_index = 0.0
        self.changed_tires_flag = 0.0
        self.refueled_flag = 0.0
        self.h_c = None
        self.is_repairing = 0.0
        self.refueled_amount = 0.0
        self.dent_severity = [0.0]*8
        self.last_lap_step = 0

        # weather_conditions = generate_weather_conditions(1)

        weather_start = self.weather_conditions[0]
        self.raining = weather_start["mRaining"]
        self.ambient_temp = weather_start["mAmbientTemp"]
        self.track_temp = weather_start["mTrackTemp"]
        self.path_wetness = weather_start["mPathWetness"]
        self.dent_severity_change = self.dent_severity_history[0]

        for i in range(8):
            
            self.dent_severity[i] += self.dent_severity_change[i]
            self.dent_severity[i] = min(self.dent_severity[i], 2.0)

        
        # self.race_complete_perc = 126.0 / self.end_et #Approxed delta for driving to start line(126s)
        
        self.curr_step = 0

        self.step_delta = 1/self.total_steps
        
        self.lap = 0
        self.num_race += 1
        
        self.impact_flag = 0.0
        # self.state = self._extract_state(self.telemetry_data[0], self.scoring_data[0])

        self.state = np.array([
                self.lap_dist,
                # self.race_complete_perc,
                self.fuel_tank_capacity,
                self.path_wetness,
                self.wheel1_wear,
                self.wheel2_wear,
                self.wheel3_wear,
                self.wheel4_wear,
                self.curr_step/self.total_steps,
                self.refueled_amount,
                # self.last_impact_et,
                self.raining,
                self.impact_flag,
                self.finish_status,
                self.in_pits,
                self.changed_tires_flag,
                self.is_repairing,
                
                self.num_penalties,
                self.dent_severity[0],
                self.dent_severity[1],
                self.dent_severity[2],
                self.dent_severity[3],
                self.dent_severity[4],
                self.dent_severity[5],
                self.dent_severity[6],
                self.dent_severity[7],
                self.laps,
                self.sector,
                self.num_pit_stops,
                self.tire_compound_index,
                self.usage_multiplier,

                self.wheel1_temp,
                self.wheel2_temp,
                self.wheel3_temp,
                self.wheel4_temp,
                self.last_impact_magnitude,
                self.ambient_temp,
                self.track_temp,
                self.end_et,
               
                # self.has_last_lap,
                # self.refueled_flag
                
            ], dtype=np.float32)

        # self.curr_window.append(self.state)

        obs = np.array([
            self.fuel_tank_capacity,
            self.path_wetness,
            self.wheel1_wear,
            self.wheel2_wear,
            self.wheel3_wear,
            self.wheel4_wear,
            self.curr_step/self.total_steps,
            self.raining,
            
            
            self.dent_severity[0],
            self.dent_severity[1],
            self.dent_severity[2],
            self.dent_severity[3],
            self.dent_severity[4],
            self.dent_severity[5],
            self.dent_severity[6],
            self.dent_severity[7],
            self.laps,
            self.num_pit_stops,
            self.tire_compound_index,
            self.usage_multiplier,

            self.wheel1_temp,
            self.wheel2_temp,
            self.wheel3_temp,
            self.wheel4_temp,
            self.ambient_temp,
            self.track_temp,
            self.end_et
            
        ], dtype=np.float32)


        return obs
    

    def compute_reward(self):
     
        reward = 0.0
        steps_this_lap = self.curr_step - self.last_lap_step
        reward += 100 + (1000 - steps_this_lap) * 0.1  # 606 is median steps per lap
        return reward

             

    def step(self,action):
        """One step of the environment's dynamics.(NOT ONE LAP)"""
        prev_sector = self.sector
        
        gap_between_tires = 0
        waiting_for_fuel = False
        repair_time = 0
        repair_weights = [20, 20, 30, 30, 60, 30, 30, 20]
        self.current_angle_radians = 0.0
        reward = 0.0
        while True:
            # reward = 0.0
            if self.fuel_tank_capacity <= 0.05:
                done = True
                reward = -500.0
                
                self.make_plots()
                self.history = []
                break

            if self.curr_step >= self.total_steps:
                done = True
                reward = 100.0 * self.lap_dist
                # reward += 50000 * self.laps / self.total_steps
                
                self.make_plots()
                self.history = []
                break
            
            # if (prev_sector == 2.0 and self.sector == 0.0) or (prev_sector == 1.0 and self.sector == 2.0) or self.finish_status == 1.0:
            if (prev_sector == 0.0 and self.sector == 1.0) or self.finish_status == 1.0:
                if self.laps == 0:
                    reward = 0.0
                else:
                    reward = self.compute_reward()
            
            if (prev_sector == 2.0 and self.sector == 0.0):
                
                

                break
            
            # possible_power_settings = [0.5,0.6,0.7,0.8,0.9,1]
            
            done = False
            data_lstm = [] # there will be data from LSTM model

            
            #Get lstm predictions
            data_lstm, self.h_c = generate_predictions(self.LSTM_model, self.state,self.scaler_minmax_X, self.scaler_robust_X, self.scaler_minmax_Y, self.scaler_robust_Y,self.h_c)

            #Get back from [sin, cos] to lap distance
            LAP_DIST_norm = (np.atan2(data_lstm[0],data_lstm[1]) + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)
            # print("LAP DIST NORM: ",LAP_DIST_norm)

            diff = LAP_DIST_norm - self.lap_dist
            if diff > 0.5:
                LAP_DIST_norm = self.lap_dist + 0.001


            data_lstm = np.hstack((LAP_DIST_norm, data_lstm[2:]))

            #Pit entry and exit lines got from track data
            pit_entry_line = 13483.0
            pit_exit_line = 390.0
            tank_capacity_max = 110.0
            self.refueled_amount = 0.0

            self.wheel1_wear = self.wheel1_wear - data_lstm[2]
            self.wheel2_wear = self.wheel2_wear - data_lstm[3]
            self.wheel3_wear = self.wheel3_wear - data_lstm[4]
            self.wheel4_wear = self.wheel4_wear - data_lstm[5]

            self.fuel_tank_capacity = self.fuel_tank_capacity - data_lstm[1]/tank_capacity_max

          

            sectors = {1: (0.0, 0.14),
                    2: (0.14, 0.56),
                    0: (0.56, 1.0)}
            threshold = 0.97
     
            
            #Check if lap ended       
            if (data_lstm[0] > 0.0) and (data_lstm[0] < 0.05) and self.curr_step > 30:
                if not self.lap_checked:
                    self.laps += 1
                    self.lap_checked = True
                    if self.laps == 1:
                        self.has_last_lap = 1.0
            else:
                self.lap_checked = False
        
           


            #Check current sector
            prev_sector = self.sector
            for sec, (start, end) in sectors.items():
                if start <= data_lstm[0] <= end:
                    self.sector = float(sec)
                    # if self.sector == 2:
                    #     reward = self.compute_reward(last_step)
                    # break
            
            


                
            
            lap_dist_max = 13623.9677734375
            pit_entry_line_dist = pit_entry_line / lap_dist_max
            pit_exit_line_dist = (pit_exit_line / lap_dist_max) - 0.0005

            
            # Check if in pits and handle pit stop actions
            pit_exit_line_dist = pit_exit_line / lap_dist_max

            
            # Check if in pits and handle pit stop actions
            if action[0] == 1 and self.laps > 1:
                # self.h_c = None  # Reset LSTM hidden and cell states
                if (pit_entry_line_dist <= data_lstm[0] <= 1.01 or 0 <= data_lstm[0] <= pit_exit_line_dist) and self.laps != 0:
                    self.in_pits = 1.0
                    if not self.checked_pit:
                        self.checked_pit = True
                    self.changed_tires_flag = 0.0
                    # self.refueled_flag = 0.0
                    # if action[1] > 0 and changed_tires_in_pit:

                else:
                    if self.in_pits == 1.0:
                        self.num_pit_stops += 1.0
                        self.in_pits = 0.0
                    self.in_pits = 0.0
                    self.checked_pit = False
                    self.pitted = False
                    self.changed_tires_flag = 0.0
                    changed_tires_in_pit = False
                
         

                pit_zone_start = 80.0 / lap_dist_max 
                in_pit_lane = (pit_entry_line_dist <= data_lstm[0] <= 1.01 or 0 <= data_lstm[0] <= pit_exit_line_dist)

                if in_pit_lane and self.laps != 1:
                    self.in_pits = 1.0
                    if not self.checked_pit:
                        self.checked_pit = True
                        # Resetujemy stan pitstopu przy wjeździe
                        self.pit_stage = 0  
                        self.pit_timer = 0
                        self.pitted = False 
                else:
                    # Wyjazd z pitu / Jazda po torze
                    if self.in_pits == 1.0:
                        self.num_pit_stops += 1.0
                    self.in_pits = 0.0
                    self.checked_pit = False
                    self.pitted = False
                    self.pit_stage = 0

                # 2. OBSŁUGA POSTOJU (Samochód stoi w miejscu)
                # Warunek: Jesteśmy w strefie pitu, mamy flagę in_pits i procedura (pitted) jeszcze nie zakończona
                if data_lstm[0] > pit_zone_start and self.in_pits == 1.0 and not self.pitted:
                    
                    #Stop impact and dent updates during pitstop
                    self.impact_flag_history[self.curr_step] = 0.0
                    self.impact_magnitude_history[self.curr_step] = 0.0
                    self.dent_severity_history[self.curr_step] = 8 * [0.0]

                    
                    if self.pit_stage == 0:
                        self.pit_snapshot = data_lstm.copy()

                    if self.pit_stage != 2.5:
                        data_lstm[1] = self.pit_snapshot[1] #constant fuel level during pitstop
                        
                    
                    # ZATRZYMANIE POJAZDU
                    data_lstm[0] = 0.006  # Ustawienie pozycji "w boksie" (upewnij się, że to nie teleportuje auta w złe miejsce!)

                    # --- FAZA 1: OPONY ---
                    if self.pit_stage == 0:
                        if action[1] > 0: # Jeśli jest żądanie zmiany opon
                            self.pit_timer = 8 # Czas trwania wymiany
                            self.tire_compound_index = action[1] - 1.0
                            self.changed_tires_flag = 1.0
                            # for i in range(2, 6):
                            #     data_lstm[i] = 1.0 # Reset zużycia opon w danych LSTM
                            self.wheel1_wear = 1.0
                            self.wheel2_wear = 1.0
                            self.wheel3_wear = 1.0
                            self.wheel4_wear = 1.0
                            self.pit_stage = 1 # Przechodzimy do wykonywania
                        else:
                            self.pit_stage = 2 # Pomijamy opony, idziemy do paliwa

                    elif self.pit_stage == 1: # Wykonywanie zmiany opon
                        if self.pit_timer > 0:
                            self.pit_timer -= 1
                        else:
                            # Koniec wymiany opon - resetujemy zużycie
                            
                            
                            self.pit_stage = 2 # Przejście do paliwa

                    # --- FAZA 2: PALIWO ---
                    elif self.pit_stage == 2:
                        self.target_fuel = action[3] * 0.2
                        current_fuel = self.fuel_tank_capacity # lub data_lstm[1] zależnie jak przechowujesz
                        self.fuel_needed = self.target_fuel - current_fuel

                        if self.fuel_needed > 0:
                            # Obliczamy czas tankowania (np. 1 litr = 1 step)
                            # Możesz też tankować "po trochu" w każdej klatce
                            # self.pit_timer = int(fuel_needed * 10) # Przykładowy przelicznik czasu
                            # Do statystyk
                            self.pit_stage = 2.5 # Wykonywanie tankowania (stan pośredni)
                        else:
                            self.pit_stage = 3 # Nie trzeba tankować, idziemy do napraw

                    elif self.pit_stage == 2.5: # Wykonywanie tankowania
                        if self.fuel_needed > 0:
                            current_fuel = self.fuel_tank_capacity
                            self.fuel_needed = self.target_fuel - current_fuel
                            # self.pit_timer -= 1
                            self.refueled_amount = max(max(1.6001, self.fuel_needed), 0) / tank_capacity_max
                            self.fuel_tank_capacity += self.refueled_amount
                            self.pit_snapshot[1] = self.fuel_tank_capacity
                        
                        else:
                            # Koniec tankowania - dolewamy całość (lub resztę)
                            # data_lstm[1] += self.refueled_amount 
                            self.pit_stage = 3 # Przejście do napraw

                    # --- FAZA 3: NAPRAWY ---
                    elif self.pit_stage == 3:
                        if action[2] == 1 and any(np.array(self.dent_severity) > 0.0):
                            # Oblicz czas naprawy TYLKO RAZ
                            total_repair_time = 0
                            for i in range(len(self.dent_severity)):
                                if self.dent_severity[i] > 0.0:
                                    total_repair_time += repair_weights[i] * self.dent_severity[i]
                            
                            self.pit_timer = total_repair_time
                            self.is_repairing = 1.0
                            self.pit_stage = 3.5 # Wykonywanie napraw
                        else:
                            self.pit_stage = 4 # Brak napraw, koniec

                    elif self.pit_stage == 3.5: # Wykonywanie napraw
                        if self.pit_timer > 0:
                            self.pit_timer -= 1
                        else:
                            # Koniec napraw - resetujemy uszkodzenia
                            self.is_repairing = 0.0
                            for i in range(len(self.dent_severity)):
                                self.dent_severity[i] = 0.0
                            self.pit_stage = 4 # Koniec

                    # --- FAZA 4: FINALIZACJA ---
                    elif self.pit_stage == 4:
                        self.pitted = True # Zwalnia blokadę, auto może ruszyć
                        data_lstm[0] += 0.0005

                # 3. POWOLNY START ("WAKE UP" LSTM)
                elif data_lstm[0] > pit_zone_start and self.in_pits == 1.0 and self.pitted:
                    # Tutaj "oszukujemy" LSTM, powoli zmieniając parametry, żeby zaczął "czuć" jazdę
                    data_lstm[0] += 0.0005  # Powolne ruszanie (zwiększanie dystansu)
                    # data_lstm[1] -= 0.001   # Symulacja zużycia paliwa
                    
        

            self.last_impact_magnitude = self.impact_magnitude_history[self.curr_step]
            self.impact_flag = self.impact_flag_history[self.curr_step]
            # self.dent_severity = self.dent_severity_history[self.curr_step]

            self.dent_severity_change = self.dent_severity_history[self.curr_step]

            for i in range(8):
                if self.in_pits == 1.0:
                    self.dent_severity_change[i] = 0.0
                
                self.dent_severity[i] += self.dent_severity_change[i]
                self.dent_severity[i] = min(self.dent_severity[i], 2.0)
                #IN SAMPLES DENT SEVERITY[0] MAX IS 1.0
                self.dent_severity[0] = min(self.dent_severity[i], 1.0)

            
            # weather_conditions = generate_weather_conditions(1,self.raining,self.ambient_temp,self.track_temp)
            weather_conditions = self.weather_conditions[self.curr_step]

            weather_start = weather_conditions
            self.raining = weather_start["mRaining"]
            self.ambient_temp = weather_start["mAmbientTemp"]
            self.track_temp = weather_start["mTrackTemp"]
            self.path_wetness = weather_start["mPathWetness"]
                            

            self.lap_dist = data_lstm[0]
            self.wheel1_temp = data_lstm[6]
            self.wheel2_temp = data_lstm[7]
            self.wheel3_temp = data_lstm[8]
            self.wheel4_temp = data_lstm[9]
            # self.path_wetness = data_lstm[10]
            self.curr_step += 1

            # if self.curr_step % 100 == 0:
            #     print(f"Step: {self.curr_step}/{self.total_steps}")

            

            # self.state = self._extract_state(self.telemetry_data[self.current_lap], self.scoring_data[self.current_lap])
            
            

            self.state = np.array([
                self.lap_dist,
                # self.race_complete_perc,
                self.fuel_tank_capacity,
                self.path_wetness,
                self.wheel1_wear,
                self.wheel2_wear,
                self.wheel3_wear,
                self.wheel4_wear,
                self.curr_step/self.total_steps,
                self.refueled_amount,
                # self.last_impact_et,
                self.raining,
                self.impact_flag,
                # self.finish_status,
                self.in_pits,
                self.changed_tires_flag,
                self.is_repairing,
                
                self.num_penalties,
                self.dent_severity[0],
                self.dent_severity[1],
                self.dent_severity[2],
                self.dent_severity[3],
                self.dent_severity[4],
                self.dent_severity[5],
                self.dent_severity[6],
                self.dent_severity[7],
                self.laps,
                self.sector,
                self.num_pit_stops,
                self.tire_compound_index,
                self.usage_multiplier,

                self.wheel1_temp,
                self.wheel2_temp,
                self.wheel3_temp,
                self.wheel4_temp,
                self.last_impact_magnitude,
                self.ambient_temp,
                self.track_temp,
                self.end_et,
               
                # self.has_last_lap,
                # self.refueled_flag
                
            ], dtype=np.float32)

            # self.curr_window.append(self.state)

            self.history.append(self.state)

            self.prev_et = data_lstm[1] * self.end_et

          



            # print(self.state)

        obs = np.array([
            self.fuel_tank_capacity,
            self.path_wetness,
            self.wheel1_wear,
            self.wheel2_wear,
            self.wheel3_wear,
            self.wheel4_wear,
            self.curr_step/self.total_steps,
            self.raining,
            
            
            self.dent_severity[0],
            self.dent_severity[1],
            self.dent_severity[2],
            self.dent_severity[3],
            self.dent_severity[4],
            self.dent_severity[5],
            self.dent_severity[6],
            self.dent_severity[7],
            self.laps,
            self.num_pit_stops,
            self.tire_compound_index,
            self.usage_multiplier,

            self.wheel1_temp,
            self.wheel2_temp,
            self.wheel3_temp,
            self.wheel4_temp,
            self.ambient_temp,
            self.track_temp,
            self.end_et
            
        ], dtype=np.float32)

        return obs, reward, done, {}



    def make_plots(self):
        history_array = np.array(self.history)
        
        # Utworzenie większej figury dla wszystkich wykresów
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Lap Distance
        plt.subplot(6, 7, 1)
        plt.plot(history_array[:, 0], label='Lap Distance', color='blue')
        plt.title('Lap Distance')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)

        # 2. Fuel Tank Capacity
        plt.subplot(6, 7, 2)
        plt.plot(history_array[:, 1], label='Fuel', color='green')
        plt.title('Fuel Tank Capacity')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)

        plt.subplot(6, 7, 3)
        plt.plot(history_array[:, 2], label='Path Wetness', color='purple')
        plt.title('Path Wetness')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)


        # 3-6. Wheel Wear (all 4 wheels)
        plt.subplot(6, 7, 4)
        plt.plot(history_array[:, 3], label='Wheel 1')
        plt.plot(history_array[:, 4], label='Wheel 2')
        plt.plot(history_array[:, 5], label='Wheel 3')
        plt.plot(history_array[:, 6], label='Wheel 4')
        plt.title('Wheel Wear')
        plt.xlabel('Time Steps')
        plt.ylabel('Wear')
        plt.legend()
        plt.grid(True)


        
        plt.subplot(6, 7, 5)
        plt.plot(history_array[:, 7], label='Step Ratio', color='orange')
        plt.title('Current Step Ratio')
        plt.xlabel('Time Steps')
        plt.ylabel('Ratio')
        plt.legend()
        plt.grid(True)

         # 13. Refueled Amount
        plt.subplot(6, 7, 6)
        plt.plot(history_array[:, 8], label='Refueled Amount', color='orange')
        plt.title('Refueled Amount')
        plt.xlabel('Time Steps')
        plt.ylabel('liters')
        plt.legend()
        plt.grid(True)

        # 16. Raining
        plt.subplot(6, 7, 7)
        plt.plot(history_array[:, 9], label='Raining', color='skyblue')
        plt.title('Raining Status')
        plt.xlabel('Time Steps')
        plt.ylabel('0/1')
        plt.legend()
        plt.grid(True)

         # 19. Impact flag
        plt.subplot(6, 7, 8)
        plt.plot(history_array[:, 10], label='Impact Flag', color='gray')
        plt.title('Impact Flag')
        plt.xlabel('Time Steps')
        plt.ylabel('Flag')
        plt.legend()
        plt.grid(True)

        # 33. In Pits
        plt.subplot(6, 7, 9)
        plt.plot(history_array[:, 11], label='In Pits', color='magenta')
        plt.title('In Pits Status')
        plt.xlabel('Time Steps')
        plt.ylabel('0/1')
        plt.legend()
        plt.grid(True)

         # 36. Changed Tires Flag
        plt.subplot(6, 7, 10)
        plt.plot(history_array[:, 12], label='Changed Tires', color='lime')
        plt.title('Changed Tires Flag')
        plt.xlabel('Time Steps')
        plt.ylabel('0/1')
        plt.legend()
        plt.grid(True)

        # 37. Is Repairing Flag
        plt.subplot(6, 7, 11)
        plt.plot(history_array[:, 13], label='Is Repairing', color='salmon')
        plt.title('Is Repairing Flag')
        plt.xlabel('Time Steps')
        plt.ylabel('0/1')
        plt.legend()
        plt.grid(True)

        # 15. Num Penalties
        plt.subplot(6, 7, 12)
        plt.plot(history_array[:, 14], label='Penalties', color='black')
        plt.title('Number of Penalties')
        plt.xlabel('Time Steps')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)

        # 20-27. Dent Severity (all 8 dents)
        plt.subplot(6, 7, 13)
        for i in range(8):
            plt.plot(history_array[:, 15 + i], label=f'Dent {i}')
        plt.title('Dent Severity')
        plt.xlabel('Time Steps')
        plt.ylabel('Severity')
        plt.legend(fontsize=6)
        plt.grid(True)

        # 30. Total Laps
        plt.subplot(6, 7, 14)
        plt.plot(history_array[:, 23], label='Total Laps', color='navy')
        plt.title('Total Laps')
        plt.xlabel('Time Steps')
        plt.ylabel('Laps')
        plt.legend()
        plt.grid(True)

       

        # 31. Sector
        plt.subplot(6, 7, 15)
        plt.plot(history_array[:, 24], label='Sector', color='green')
        plt.title('Sector')
        plt.xlabel('Time Steps')
        plt.ylabel('Sector (0/1/2)')
        plt.legend()
        plt.grid(True)

          # 32. Num Pitstops
        plt.subplot(6, 7, 16)
        plt.plot(history_array[:, 25], label='Num Pitstops', color='olive')
        plt.title('Number of Pitstops')
        plt.xlabel('Time Steps')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)

         # 34. Tire Compound Index
        plt.subplot(6, 7, 17)
        plt.plot(history_array[:, 26], label='Tire Compound', color='teal')
        plt.title('Tire Compound Index')
        plt.xlabel('Time Steps')
        plt.ylabel('Index')
        plt.legend()
        plt.grid(True)

         # 35. Usage Multiplier
        plt.subplot(6, 7, 18)
        plt.plot(history_array[:, 27], label='Usage Multiplier', color='coral')
        plt.title('Usage Multiplier')
        plt.xlabel('Time Steps')
        plt.ylabel('Multiplier')
        plt.legend()
        plt.grid(True)


        # 7-10. Wheel Temperature (all 4 wheels)
        plt.subplot(6, 7, 19)
        plt.plot(history_array[:, 28], label='Wheel 1')
        plt.plot(history_array[:, 29], label='Wheel 2')
        plt.plot(history_array[:, 30], label='Wheel 3')
        plt.plot(history_array[:, 31], label='Wheel 4')
        plt.title('Wheel Temperature')
        plt.xlabel('Time Steps')
        plt.ylabel('Temp (°C)')
        plt.legend()
        plt.grid(True)

        # 11. Path Wetness
        

        # 12. Current Step Ratio
        

       

        # 14. Last Impact Magnitude
        plt.subplot(6, 7, 20)
        plt.plot(history_array[:, 32], label='Impact Magnitude', color='darkred')
        plt.title('Last Impact Magnitude')
        plt.xlabel('Time Steps')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.grid(True)

        

        

        # 17. Ambient Temperature
        plt.subplot(6, 7, 21)
        plt.plot(history_array[:, 33], label='Ambient Temp', color='brown')
        plt.title('Ambient Temperature')
        plt.xlabel('Time Steps')
        plt.ylabel('Temp (°C)')
        plt.legend()
        plt.grid(True)

        # 18. Track Temperature
        plt.subplot(6, 7, 22)
        plt.plot(history_array[:, 34], label='Track Temp', color='cyan')
        plt.title('Track Temperature')
        plt.xlabel('Time Steps')
        plt.ylabel('Temp (°C)')
        plt.legend()
        plt.grid(True)

        # 19. End ET
        plt.subplot(6, 7, 23)
        plt.plot(history_array[:, 35], label='End ET', color='gray')
        plt.title('End ET')
        plt.xlabel('Time Steps')
        plt.ylabel('ET')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'ai/rl_training_race_historyplots/race_history_plots_{self.num_race}.png', dpi=150)
        # plt.show()
        plt.close(fig)
# ...existing code...
        


