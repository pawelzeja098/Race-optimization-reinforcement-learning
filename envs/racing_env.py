import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json

matplotlib.use('TkAgg')

import gymnasium as gym
from gymnasium import spaces

data_race_scoring = "data/scoring_data.json"
data_race_telemetry = "data/telemetry_data.json"

class RacingEnv(gym.Env):
    def __init__(self):
        super(RacingEnv,self).__init__()

        self.lap = 0
        self.usage_multiplier = 1.0
        

        with open(data_race_scoring, "r") as file:
            self.scoring_data = json.load(file)
        with open(data_race_telemetry, "r") as file:
            self.telemetry_data = json.load(file)
        
        

        #tires - 0 - soft, 1 - medium, 2 - hard , 3 - wet 
        # self.state = [1, 1, 1, 1, 1, 0, 0.8] #Engine, suspension, brakes,fuel, tires_wear tires_type, car_power
        self.state = self._extract_state(self.telemetry_data[0], self.scoring_data[0])
        
        
        self.observation_space = gym.spaces.Box(
    low=np.array([
        0.0,   # mLastLapTime (zakładamy nieujemny)
        0.0,   # mHasLastLap (bool, ale jako float 0/1)
        0.0,   # mBestLapTime
        0.0,   # mCurrLapTime
        0.0,   # mInPits (bool, ale jako float 0/1)
        0.0,   # mNumPitstops
        0.0,   # mRaining
        -50.0, # mAmbientTemp (np. -50 do +60 C)
        -50.0, # mTrackTemp
        0.0,   # mEndET
        0.0,   # mCurrentET
        0.0,   # mFuel
        0.0,   # mFuelCapacity
        0.0,   # avg tire wear
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # dent severities
        0.0,   # tire compound index
    ], dtype=np.float32),
    high=np.array([
        10000.0, # mLastLapTime (max np. 10000 s)
        1.0,     # mHasLastLap (bool, ale jako float 0/1)
        10000.0, # mBestLapTime
        10000.0, # mCurrLapTime
        1.0,     # mInPits
        50.0,    # mNumPitstops (duży limit)
        1.0,     # mRaining
        60.0,    # mAmbientTemp
        100.0,   # mTrackTemp
        100000.0,# mEndET
        100000.0,# mCurrentET
        200.0,   # mFuel
        200.0,   # mFuelCapacity
        1.0,     # avg tire wear (0-1)
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,  # dent severities
        3.0,    # tire compound index
    ], dtype=np.float32),
    dtype=np.float32
)
    
        

        self.action_space = spaces.MultiDiscrete([
                                                2, # Pit stop or not
                                                2, # Confirm pit stop or not
                                                5, # Tire change (0-4) No, soft, medium, hard, wet
                                                2, # Repair or not (0-1)
                                                23, # Fuel * 5l (0-22)
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


    

    def step(self,action):
        # possible_power_settings = [0.5,0.6,0.7,0.8,0.9,1]
        self.state = self._extract_state(self.telemetry_data[self.current_lap], self.scoring_data[self.current_lap])
        
        self.current_lap += 1


     