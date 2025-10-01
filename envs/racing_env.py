import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json

matplotlib.use('TkAgg')

import gymnasium as gym
from gymnasium import spaces
from weather_generator import generate_weather_conditions
from impact_generator import random_impact_magnitude, generate_dent_severity

data_race_scoring = "data/scoring_data.json"
data_race_telemetry = "data/telemetry_data.json"

class RacingEnv(gym.Env):
    def __init__(self):
        super(RacingEnv,self).__init__()

        self.lap = 0
        self.usage_multiplier = 1.0
        self.checked_pit = False
        

        with open(data_race_scoring, "r") as file:
            self.scoring_data = json.load(file)
        with open(data_race_telemetry, "r") as file:
            self.telemetry_data = json.load(file)
        
        

        #tires - 0 - soft, 1 - medium, 2 - hard , 3 - wet 
        # self.state = [1, 1, 1, 1, 1, 0, 0.8] #Engine, suspension, brakes,fuel, tires_wear tires_type, car_power
        self.state = self._extract_state(self.telemetry_data[0], self.scoring_data[0])
        
        
        self.observation_space = gym.spaces.Box(
    low=np.array([
        0.0,   # Lap Dist
        0.0,   # Race complete %
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
        0.0,   # Last impact ET
        0.0,   # Last impact magnitude
        0.0,   # Number of penalties
        0.0,   # Raining
        0.0,   # Ambient temp
        0.0,   # Track temp
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # dent severities
        0.0,   # #has last lap
        0.0,   # Finish status
        0.0,   # Total laps
        0.0,   # Sector
        0.0,   # Num pitstops
        0.0,   # In pits
        0.0,   # tire compound index
        1.0    # multiplier

    ], dtype=np.float32),
    high=np.array([
        1,2,     # Lap Dist
        2.0,     # Race complete %
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
        86500.0,   # Last impact ET
        50000.0,     # Last impact magnitude
        100.0,   # Number of penalties
        1.0,   # Raining
        45.0,   # Ambient temp
        60.0,   # Track temp
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,  # dent severities
        1.0,   # #has last lap
        1.0,   # Finish status
        500.0,   # Total laps
        2.0,   # Sector
        100.0,   # Num pitstops
        1.0,   # In pits
        3.0,    # tire compound index
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
        
    def _extract_state(self, telemetry, scoring):
            wear = 0
            for wheel in telemetry["mWheel"]:
                # Process each wheel's telemetry data
                #for now only wear
                wear += wheel["mWear"]
            
            last_lap = scoring["mLastLapTime"]
            has_last_lap = 1.0 if last_lap > 0 else 0.0
            if last_lap <= 0.0:
                last_lap = 10000.0  # placeholder
            
                

            return np.array([
                last_lap,
                has_last_lap,
                scoring["mBestLapTime"],
                # scoring["mCurrLapTime"],
                int(scoring["mInPits"]),
                scoring["mNumPitstops"],
                scoring["mRaining"],
                round(scoring["mAmbientTemp"], 2),
                round(scoring["mTrackTemp"], 2),
                scoring["mEndET"],
                scoring["mCurrentET"],

                round(telemetry["mFuel"],2),
                round(telemetry["mFuelCapacity"],2),
                round(wear / 4.0, 2),  # Average wear across all four tires
                telemetry["mDentSeverity"][0],  # Not defined which part of the car this refers to each index
                telemetry["mDentSeverity"][1],
                telemetry["mDentSeverity"][2], 
                telemetry["mDentSeverity"][3],
                telemetry["mDentSeverity"][4],
                telemetry["mDentSeverity"][5],
                telemetry["mDentSeverity"][6], 
                telemetry["mDentSeverity"][7],

                telemetry["mFrontTireCompoundIndex"],
                self.usage_multiplier
                
        ], dtype=np.float32)
    


    def reset(self):
       
        self.lap = 0
        self.usage_multiplier = 1.0
        self.state = self._extract_state(self.telemetry_data[0], self.scoring_data[0])
        return self.state

    
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


    def step(self,action):
        # possible_power_settings = [0.5,0.6,0.7,0.8,0.9,1]
        data = [] # there will be data from LSTM model

        pit_entry_line = 13483.0
        pit_exit_line = 390.0
        

        sectors = {1: (0.0, 0.14),
                   0: (0.14, 0.56),
                   2: (0.56, 1.0)}
        threshold = 0.97
        
        #Check if lap ended       
        if data[:-2][0] > threshold and data[:-1][0] < 0.1:
            laps += 1

        #Check if race is finished
        if data[:-1][1] >= 1.0 and data[:-1][0] > threshold:
            finish_status = 1.0

            
        action = []# tutaj będzie akcja z modelu BC
        lap_dist_max = 13623.9677734375
        pit_entry_line_dist = pit_entry_line / lap_dist_max
        pit_exit_line_dist = pit_exit_line / lap_dist_max
        
        if action[0] == 1:
            if data[:-1][0] >= pit_entry_line_dist and data[:-1][0] <= pit_exit_line_dist:
                in_pits = 1.0
                if not self.checked_pit:
                    self.number_of_pit_stops += 1.0
                    self.checked_pit = True
            else:
                in_pits = 0.0
                self.checked_pit = False
                
            if data[:-1][0] > 80.0:
                tire_index = action[1] - 1.0
        


        self.state = self._extract_state(self.telemetry_data[self.current_lap], self.scoring_data[self.current_lap])
        
        self.current_lap += 1


#      Sektor zmienił się na 1.0 przy dystansie -0.01213
# Sektor zmienił się na 2.0 przy dystansie 0.13955
# Sektor zmienił się na 0.0 przy dystansie 0.5677
# Sektor zmienił się na 1.0 przy dystansie 0.00034
# Sektor zmienił się na 2.0 przy dystansie 0.14061
# Sektor zmienił się na 0.0 przy dystansie 0.56768
# Sektor zmienił się na 1.0 przy dystansie 0.00111