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

data_race_scoring = "data/scoring_data.json"
data_race_telemetry = "data/telemetry_data.json"

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
        
        self.laps = 0
        self.checked_pit = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state = None
        self.scaler_X = joblib.load("models/scaler1_X.pkl")
        self.scaler_Y = joblib.load("models/scaler1_Y.pkl")
        self.h_c = None
        self.lap_checked = False

        self.LSTM_model = LSTMStatePredictor(input_size=38, hidden_size=256, output_size=12, num_layers=1).to(device)
        self.LSTM_model.load_state_dict(torch.load("models/lstm1_model.pth", map_location=device))
        self.LSTM_model.eval()
        # self.curr_window = deque(maxlen=30)

        

        # with open(data_race_scoring, "r") as file:
        #     self.scoring_data = json.load(file)
        # with open(data_race_telemetry, "r") as file:
        #     self.telemetry_data = json.load(file)
        
        

        #tires - 0 - soft, 1 - medium, 2 - hard , 3 - wet 
        # self.state = [1, 1, 1, 1, 1, 0, 0.8] #Engine, suspension, brakes,fuel, tires_wear tires_type, car_power
        # self.state = self._extract_state(self.telemetry_data[0], self.scoring_data[0])
        
        
        self.observation_space = gym.spaces.Box(
    low=np.array([
        0.0,   # Lap Dist
        # 0.0,   # Race complete %
        0.0,   # Tank capacity
        0.0,   # Wheel wear
        0.0,   # Wheel wear
        0.0,   # Wheel wear
        0.0,   # Wheel wear
        0.0,   # Wheel temperature
        0.0,   # Wheel temperature
        0.0,   # Wheel temperature
        0.0,   # Wheel temperature
        0.0,   # Path wetness
        0.0,   # Current step ratio
        0.0,   # Last impact ET
        0.0,   # Last impact magnitude
        0.0,   # Number of penalties
        0.0,   # Raining
        0.0,   # Ambient temp
        0.0,   # Track temp
        0.0,   # End ET
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # dent severities
        0.0,   # #has last lap
        0.0,   # Finish status
        0.0,   # Total laps
        0.0,   # Sector
        0.0,   # Num pitstops
        0.0,   # In pits
        0.0,   # tire compound index
        0.0,   # changed tires flag
        0.0,   # refueled flag
        1.0    # multiplier

    ], dtype=np.float32),
    high=np.array([
        1.2,     # Lap Dist
        # 2.0,     # Race complete %
        1.1,     # Tank capacity
        1.0,     # Wheel wear
        1.0,     # Wheel wear
        1.0,     # Wheel wear
        1.0,     # Wheel wear
        600.0,   # Wheel temperature
        600.0,   # Wheel temperature
        600.0,   # Wheel temperature
        600.0,   # Wheel temperature
        1.0,    # Path wetness
        1.0,    # Current step ratio
        86500.0,   # Last impact ET
        50000.0,     # Last impact magnitude
        100.0,   # Number of penalties
        1.0,   # Raining
        45.0,   # Ambient temp
        60.0,   # Track temp
        86500.0,   # End ET
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,  # dent severities
        1.0,   # #has last lap
        1.0,   # Finish status
        500.0,   # Total laps
        2.0,   # Sector
        100.0,   # Num pitstops
        1.0,   # In pits
        3.0,    # tire compound index
        1.0,    # changed tires flag
        1.0,    # refueled flag
        1.0    # multiplier
    ], dtype=np.float32),
    dtype=np.float32
)
    
        

        self.action_space = spaces.MultiDiscrete([
                                                2, # Pit stop or not
                                                # 2, # Confirm pit stop or not
                                                5, # Tire change (0-4) No, soft, medium, hard, wet
                                                2, # Repair or not (0-1)
                                                21, # Fuel * 0.05 (0-20)
                                                ])
        
   


    def reset(self):
        self.laps = 0
        
        self.lap_dist = 0.0
        # self.race_complete_perc = 0.0
        self.fuel_tank_capacity = 1.0
        self.wheel1_wear = 1.0
        self.wheel2_wear = 1.0
        self.wheel3_wear = 1.0
        self.wheel4_wear = 1.0
        self.wheel1_temp = 340.0
        self.wheel2_temp = 340.0
        self.wheel3_temp = 340.0
        self.wheel4_temp = 340.0
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
        self.laps = 0.0
        self.sector = 0.0
        self.num_pit_stops = 0.0
        self.in_pits = 0.0
        self.tire_compound_index = 3.0
        self.changed_tires_flag = 0.0
        self.refueled_flag = 0.0

        weather_conditions = generate_weather_conditions(1)

        weather_start = weather_conditions
        self.raining = weather_start["mRaining"]
        self.ambient_temp = weather_start["mAmbientTemp"]
        self.track_temp = weather_start["mTrackTemp"]

        self.end_et = 734.0
        self.race_complete_perc = 126.0 / self.end_et #Approxed delta for driving to start line(126s)
        self.total_steps = 1600
        self.curr_step = 0

        self.step_delta = 1/self.total_steps
        
        self.lap = 0
        self.usage_multiplier = 1.0
        # self.state = self._extract_state(self.telemetry_data[0], self.scoring_data[0])

        self.state = np.array([
            # self.lap_dist,
            self.race_complete_perc,
            self.fuel_tank_capacity,
            self.wheel1_wear,
            self.wheel2_wear,
            self.wheel3_wear,
            self.wheel4_wear,
            self.wheel1_temp,
            self.wheel2_temp,
            self.wheel3_temp,
            self.wheel4_temp,
            self.path_wetness,
            self.curr_step/self.total_steps,
            self.last_impact_et,
            self.last_impact_magnitude,
            self.num_penalties,
            self.raining,
            self.ambient_temp,
            self.track_temp,
            self.end_et,
            self.dent_severity[0],
            self.dent_severity[1],
            self.dent_severity[2],
            self.dent_severity[3],
            self.dent_severity[4],
            self.dent_severity[5],
            self.dent_severity[6],
            self.dent_severity[7],
            self.has_last_lap,
            self.finish_status,
            self.laps,
            self.sector,
            self.num_pit_stops,
            self.in_pits,
            self.tire_compound_index,
            self.usage_multiplier,
            self.changed_tires_flag,
            self.refueled_flag
        ], dtype=np.float32)

        # self.curr_window.append(self.state)


        return self.state
    
    def start_configuration(self,action):
        self.tire_compound_index = action[1] - 1.0
        self.fuel_tank_capacity = max(action[3] * 0.05, 1.0)
        self.wheel1_wear = 1.0
        self.wheel2_wear = 1.0
        self.wheel3_wear = 1.0
        self.wheel4_wear = 1.0
        self.wheel1_temp = 300.0
        self.wheel2_temp = 300.0
        self.wheel3_temp = 300.0
        self.wheel4_temp = 300.0



    
#       mLastImpactET: min=0.0, max=844.02001953125 mam
# mLastImpactMagnitude: min=0.0, max=22007.326171875 mam
# mNumPenalties: min=0.0, max=1.0 nie mam, narazie nie biorę pod uwagę
# mRaining: min=0.0, max=1.0 mam
# mAmbientTemp: min=5.33, max=40.0 mam
# mTrackTemp: min=9.0, max=47.35 mam
# mDentSeverity[0]: min=0.0, max=2.0 mam
# mDentSeverity[1]: min=0.0, max=2.0 mam
# mDentSeverity[2]: min=0.0, max=0.0 mam
# mDentSeverity[3]: min=0.0, max=2.0 mam
# mDentSeverity[4]: min=0.0, max=2.0 mam
# mDentSeverity[5]: min=0.0, max=2.0 mam
# mDentSeverity[6]: min=0.0, max=0.0 mam
# mDentSeverity[7]: min=0.0, max=2.0 mam
# has_last_lap: min=0.0, max=1.0 nie mam
# mFinishStatus: min=0.0, max=1.0 mam
# mTotalLaps: min=0.0, max=9.0 mam
# mSector: min=0.0, max=2.0 mam
# mNumPitstops: min=0.0, max=2.0 nie mam
# mInPits: min=0.0, max=1.0 mam
# mFrontTireCompoundIndex: min=0.0, max=3.0 nie mam 
# multiplier: min=1.0, max=3.0 nie mam
# Feature 22: min=0.0, max=404.0758056640625
# Feature 23: min=0.0, max=354.8138427734375
#Do RL moze bym jeszcze potrzebowal całkowity race time

    def compute_reward(self,last_step):
        reward = 0.0
        if self.sector == 0.0:
            lap_time = self.end_et * self.race_complete_perc
        elif self.sector == 2.0:
            delta = self.race_complete_perc/self.lap - last_step[0]
            reward += delta * 100.0
            if self.fuel_tank_capacity <= 0.05:
                reward -= 50.0
            if self.wheel1_wear >= 0.2 or self.wheel2_wear >= 0.2 or self.wheel3_wear >= 0.2 or self.wheel4_wear >= 0.2:
                reward -= 20.0
        elif self.finish_status == 1.0:
            reward += 500.0
            reward += 5 * self.laps
        return reward
        


        
        

    def step(self,action,last_step):
        prev_sector = self.sector
        while True:
            if (prev_sector == 2.0 and self.sector == 0.0) or (prev_sector == 1.0 and self.sector == 2.0) or self.finish_status == 1.0:
                if self.lap == 0:
                    reward = 0.0
                else:
                    reward = self.compute_reward(last_step)

                
                
                break
            
            # possible_power_settings = [0.5,0.6,0.7,0.8,0.9,1]
            
            done = False
            data_lstm = [] # there will be data from LSTM model

            

            data_lstm, self.h_c = generate_predictions(self.LSTM_model, self.state,self.scaler_X, self.scaler_Y,self.h_c)

            LAP_DIST_norm = (np.atan2(data_lstm[0],data_lstm[1]) + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)

            data_lstm = np.hstack((LAP_DIST_norm, data_lstm[2:]))

            pit_entry_line = 13483.0
            pit_exit_line = 390.0

            # if data_lstm[1] < 0.0:
            #     data_lstm[1] = 0.0

            for i in range(2, 6):
                if data_lstm[i] > 1.0:
                    data_lstm[i] = 1.0
            
            if data_lstm[10] < 0.0:
                data_lstm[10] = 0.0

            sectors = {1: (0.0, 0.14),
                    2: (0.14, 0.56),
                    0: (0.56, 1.0)}
            threshold = 0.97
            # print("LAP DIST: ",data_lstm[0])
            # print("Race DIST: ",data_lstm[1])
            
            #Check if lap ended       
            if (data_lstm[0] > 0.0) and (data_lstm[0] < 0.05):
                if not self.lap_checked:
                    self.laps += 1
                    self.lap_checked = True
                    if self.laps == 1:
                        self.has_last_lap = 1.0
            else:
                self.lap_checked = False
            #Check if race is finished
            # if data_lstm[1] >= 1.0 and data_lstm[0] > threshold:
            #     finish_status = 1.0
            #     if finish_status == 1.0:
            #         # reward = 1000.0
            #         done = True
            #         pass
            if self.curr_step >= self.total_steps:
                done = True

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
            pit_exit_line_dist = pit_exit_line / lap_dist_max
            
            # Check if in pits and handle pit stop actions
            if action[0] == 1:
                if pit_entry_line_dist <= data_lstm[0] <= 1.01 and 0 <= data_lstm[0] <= pit_exit_line_dist and self.laps != 0:
                    self.in_pits = 1.0
                    if not self.checked_pit:
                        self.number_of_pit_stops += 1.0
                        self.checked_pit = True
                else:
                    self.in_pits = 0.0
                    self.checked_pit = False
                    
                if data_lstm[0] > 80.0:
                    self.tire_compound_index = action[1] - 1.0
                    self.changed_tires_flag = 1.0
                    self.fuel_tank_capacity = action[3] * 0.05
                    self.refueled_flag = 1.0
                    if action[2] == 1:
                        for i in range(len(self.dent_severity)):
                            self.dent_severity[i] = 0.0
                
            
            impact_magnitude = random_impact_magnitude()
            if impact_magnitude > 0.0:
                self.last_impact_et = abs(data_lstm[1] * self.end_et  - self.prev_et)
                
            self.dent_severity = generate_dent_severity(impact_magnitude,self.dent_severity)

            weather_conditions = generate_weather_conditions(1,self.raining,self.ambient_temp,self.track_temp)

            weather_start = weather_conditions
            self.raining = weather_start["mRaining"]
            self.ambient_temp = weather_start["mAmbientTemp"]
            self.track_temp = weather_start["mTrackTemp"]
                            

            self.lap_dist = data_lstm[0]
            # self.race_complete_perc = data_lstm[1]
            self.fuel_tank_capacity = data_lstm[1]
            self.wheel1_wear = data_lstm[2]
            self.wheel2_wear = data_lstm[3]
            self.wheel3_wear = data_lstm[4]
            self.wheel4_wear = data_lstm[5]
            self.wheel1_temp = data_lstm[6]
            self.wheel2_temp = data_lstm[7]
            self.wheel3_temp = data_lstm[8]
            self.wheel4_temp = data_lstm[9]
            self.path_wetness = data_lstm[10]
            self.curr_step += 1

            

            # self.state = self._extract_state(self.telemetry_data[self.current_lap], self.scoring_data[self.current_lap])
            
            

            self.state = np.array([
                self.lap_dist,
                # self.race_complete_perc,
                self.fuel_tank_capacity,
                self.wheel1_wear,
                self.wheel2_wear,
                self.wheel3_wear,
                self.wheel4_wear,
                self.wheel1_temp,
                self.wheel2_temp,
                self.wheel3_temp,
                self.wheel4_temp,
                self.path_wetness,
                self.curr_step/self.total_steps,
                self.last_impact_et,
                self.last_impact_magnitude,
                self.num_penalties,
                self.raining,
                self.ambient_temp,
                self.track_temp,
                self.end_et,
                self.dent_severity[0],
                self.dent_severity[1],
                self.dent_severity[2],
                self.dent_severity[3],
                self.dent_severity[4],
                self.dent_severity[5],
                self.dent_severity[6],
                self.dent_severity[7],
                self.has_last_lap,
                self.finish_status,
                self.laps,
                self.sector,
                self.num_pit_stops,
                self.in_pits,
                self.tire_compound_index,
                self.usage_multiplier,
                self.changed_tires_flag,
                self.refueled_flag
            ], dtype=np.float32)

            # self.curr_window.append(self.state)

            self.history.append(self.state)

            self.prev_et = data_lstm[1] * self.end_et

            print("Lap dist:", self.state[0])
            # print("Race complete perc:", self.state[1])
            print("Fuel tank capacity:", self.state[1])
            print("Wheel wear:", self.state[2:6])
            print("Wheel temp:", self.state[6:10])
            print("Path wetness:", self.state[10])
            print("Current step ratio:", self.state[11])
            print("Last impact ET:", self.state[12])
            print("Last impact magnitude:", self.state[13])
            print("Num penalties:", self.state[14])
            print("Raining:", self.state[15])
            print("Ambient temp:", self.state[16])
            print("Track temp:", self.state[17])
            print("End ET:", self.state[18])
            print("Dent severity:", self.state[19:27])
            print("Has last lap:", self.state[27])
            print("Finish status:", self.state[28])
            print("Total laps:", self.state[29])
            print("Sector:", self.state[30])
            print("Num pitstops:", self.state[31])
            print("In pits:", self.state[32])
            print("Tire compound index:", self.state[33])
            print("Usage multiplier:", self.state[34])
            print("Changed tires flag:", self.state[35])
            print("Refueled flag:", self.state[36])

            print("-----")

            if len(self.history) == 1600:
                self.make_plots()
                self.history = []
                exit()



            # print(self.state)

        return self.state, reward, done, {}


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

        # 3-6. Wheel Wear (all 4 wheels)
        plt.subplot(6, 7, 3)
        plt.plot(history_array[:, 2], label='Wheel 1')
        plt.plot(history_array[:, 3], label='Wheel 2')
        plt.plot(history_array[:, 4], label='Wheel 3')
        plt.plot(history_array[:, 5], label='Wheel 4')
        plt.title('Wheel Wear')
        plt.xlabel('Time Steps')
        plt.ylabel('Wear')
        plt.legend()
        plt.grid(True)

        # 7-10. Wheel Temperature (all 4 wheels)
        plt.subplot(6, 7, 4)
        plt.plot(history_array[:, 6], label='Wheel 1')
        plt.plot(history_array[:, 7], label='Wheel 2')
        plt.plot(history_array[:, 8], label='Wheel 3')
        plt.plot(history_array[:, 9], label='Wheel 4')
        plt.title('Wheel Temperature')
        plt.xlabel('Time Steps')
        plt.ylabel('Temp (°C)')
        plt.legend()
        plt.grid(True)

        # 11. Path Wetness
        plt.subplot(6, 7, 5)
        plt.plot(history_array[:, 10], label='Path Wetness', color='purple')
        plt.title('Path Wetness')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)

        # 12. Current Step Ratio
        plt.subplot(6, 7, 6)
        plt.plot(history_array[:, 11], label='Step Ratio', color='orange')
        plt.title('Current Step Ratio')
        plt.xlabel('Time Steps')
        plt.ylabel('Ratio')
        plt.legend()
        plt.grid(True)

        # 13. Last Impact ET
        plt.subplot(6, 7, 7)
        plt.plot(history_array[:, 12], label='Impact ET', color='red')
        plt.title('Last Impact ET')
        plt.xlabel('Time Steps')
        plt.ylabel('ET')
        plt.legend()
        plt.grid(True)

        # 14. Last Impact Magnitude
        plt.subplot(6, 7, 8)
        plt.plot(history_array[:, 13], label='Impact Magnitude', color='darkred')
        plt.title('Last Impact Magnitude')
        plt.xlabel('Time Steps')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.grid(True)

        # 15. Num Penalties
        plt.subplot(6, 7, 9)
        plt.plot(history_array[:, 14], label='Penalties', color='black')
        plt.title('Number of Penalties')
        plt.xlabel('Time Steps')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)

        # 16. Raining
        plt.subplot(6, 7, 10)
        plt.plot(history_array[:, 15], label='Raining', color='skyblue')
        plt.title('Raining Status')
        plt.xlabel('Time Steps')
        plt.ylabel('0/1')
        plt.legend()
        plt.grid(True)

        # 17. Ambient Temperature
        plt.subplot(6, 7, 11)
        plt.plot(history_array[:, 16], label='Ambient Temp', color='brown')
        plt.title('Ambient Temperature')
        plt.xlabel('Time Steps')
        plt.ylabel('Temp (°C)')
        plt.legend()
        plt.grid(True)

        # 18. Track Temperature
        plt.subplot(6, 7, 12)
        plt.plot(history_array[:, 17], label='Track Temp', color='cyan')
        plt.title('Track Temperature')
        plt.xlabel('Time Steps')
        plt.ylabel('Temp (°C)')
        plt.legend()
        plt.grid(True)

        # 19. End ET
        plt.subplot(6, 7, 13)
        plt.plot(history_array[:, 18], label='End ET', color='gray')
        plt.title('End ET')
        plt.xlabel('Time Steps')
        plt.ylabel('ET')
        plt.legend()
        plt.grid(True)

        # 20-27. Dent Severity (all 8 dents)
        plt.subplot(6, 7, 14)
        for i in range(8):
            plt.plot(history_array[:, 19 + i], label=f'Dent {i}')
        plt.title('Dent Severity')
        plt.xlabel('Time Steps')
        plt.ylabel('Severity')
        plt.legend(fontsize=6)
        plt.grid(True)

        # 28. Has Last Lap
        plt.subplot(6, 7, 15)
        plt.plot(history_array[:, 27], label='Has Last Lap', color='pink')
        plt.title('Has Last Lap')
        plt.xlabel('Time Steps')
        plt.ylabel('0/1')
        plt.legend()
        plt.grid(True)

        # 29. Finish Status
        plt.subplot(6, 7, 16)
        plt.plot(history_array[:, 28], label='Finish Status', color='gold')
        plt.title('Finish Status')
        plt.xlabel('Time Steps')
        plt.ylabel('0/1')
        plt.legend()
        plt.grid(True)

        # 30. Total Laps
        plt.subplot(6, 7, 17)
        plt.plot(history_array[:, 29], label='Total Laps', color='navy')
        plt.title('Total Laps')
        plt.xlabel('Time Steps')
        plt.ylabel('Laps')
        plt.legend()
        plt.grid(True)

        # 31. Sector
        plt.subplot(6, 7, 18)
        plt.plot(history_array[:, 30], label='Sector', color='green')
        plt.title('Sector')
        plt.xlabel('Time Steps')
        plt.ylabel('Sector (0/1/2)')
        plt.legend()
        plt.grid(True)

        # 32. Num Pitstops
        plt.subplot(6, 7, 19)
        plt.plot(history_array[:, 31], label='Num Pitstops', color='olive')
        plt.title('Number of Pitstops')
        plt.xlabel('Time Steps')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)

        # 33. In Pits
        plt.subplot(6, 7, 20)
        plt.plot(history_array[:, 32], label='In Pits', color='magenta')
        plt.title('In Pits Status')
        plt.xlabel('Time Steps')
        plt.ylabel('0/1')
        plt.legend()
        plt.grid(True)

        # 34. Tire Compound Index
        plt.subplot(6, 7, 21)
        plt.plot(history_array[:, 33], label='Tire Compound', color='teal')
        plt.title('Tire Compound Index')
        plt.xlabel('Time Steps')
        plt.ylabel('Index')
        plt.legend()
        plt.grid(True)

        # 35. Usage Multiplier
        plt.subplot(6, 7, 22)
        plt.plot(history_array[:, 34], label='Usage Multiplier', color='coral')
        plt.title('Usage Multiplier')
        plt.xlabel('Time Steps')
        plt.ylabel('Multiplier')
        plt.legend()
        plt.grid(True)

        # 36. Changed Tires Flag
        plt.subplot(6, 7, 23)
        plt.plot(history_array[:, 35], label='Changed Tires', color='lime')
        plt.title('Changed Tires Flag')
        plt.xlabel('Time Steps')
        plt.ylabel('0/1')
        plt.legend()
        plt.grid(True)

        # 37. Refueled Flag
        plt.subplot(6, 7, 24)
        plt.plot(history_array[:, 36], label='Refueled', color='salmon')
        plt.title('Refueled Flag')
        plt.xlabel('Time Steps')
        plt.ylabel('0/1')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('race_history_plots.png', dpi=150)
        plt.show()



#      Sektor zmienił się na 1.0 przy dystansie -0.01213
# Sektor zmienił się na 2.0 przy dystansie 0.13955
# Sektor zmienił się na 0.0 przy dystansie 0.5677
# Sektor zmienił się na 1.0 przy dystansie 0.00034
# Sektor zmienił się na 2.0 przy dystansie 0.14061
# Sektor zmienił się na 0.0 przy dystansie 0.56768
# Sektor zmienił się na 1.0 przy dystansie 0.00111

 # def _extract_state(self, telemetry, scoring):
    #         wear = 0
    #         for wheel in telemetry["mWheel"]:
    #             # Process each wheel's telemetry data
    #             #for now only wear
    #             wear += wheel["mWear"]
            
    #         last_lap = scoring["mLastLapTime"]
    #         has_last_lap = 1.0 if last_lap > 0 else 0.0
    #         if last_lap <= 0.0:
    #             last_lap = 10000.0  # placeholder
            
                

    #         return np.array([
    #             last_lap,
    #             has_last_lap,
    #             scoring["mBestLapTime"],
    #             # scoring["mCurrLapTime"],
    #             int(scoring["mInPits"]),
    #             scoring["mNumPitstops"],
    #             scoring["mRaining"],
    #             round(scoring["mAmbientTemp"], 2),
    #             round(scoring["mTrackTemp"], 2),
    #             scoring["mEndET"],
    #             scoring["mCurrentET"],

    #             round(telemetry["mFuel"],2),
    #             round(telemetry["mFuelCapacity"],2),
    #             round(wear / 4.0, 2),  # Average wear across all four tires
    #             telemetry["mDentSeverity"][0],  # Not defined which part of the car this refers to each index
    #             telemetry["mDentSeverity"][1],
    #             telemetry["mDentSeverity"][2], 
    #             telemetry["mDentSeverity"][3],
    #             telemetry["mDentSeverity"][4],
    #             telemetry["mDentSeverity"][5],
    #             telemetry["mDentSeverity"][6], 
    #             telemetry["mDentSeverity"][7],

    #             telemetry["mFrontTireCompoundIndex"],
    #             self.usage_multiplier
                
    #     ], dtype=np.float32)
    