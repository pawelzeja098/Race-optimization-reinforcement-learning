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

        

        with open(data_race_scoring, "r") as file:
            scoring_data = json.load(file)
        with open(data_race_telemetry, "r") as file:
            telemetry_data = json.load(file)
        
        

        #tires - 0 - soft, 1 - medium, 2 - hard , 3 - wet 
        # self.state = [1, 1, 1, 1, 1, 0, 0.8] #Engine, suspension, brakes,fuel, tires_wear tires_type, car_power
        self.state = [
        
        ]
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0.5,0,0,0,0,0]),
                                    high=np.array([1, 1, 1, 1, 1, 3, 1,20,1,1,1,1000]),
                                    dtype=np.float32)
        self.current_lap = 0
        self.current_state = 0

        self.lap_time = 0



        self.action_space = gym.spaces.MultiDiscrete([2,2,2,4,2,2,2,2,5]) #nothing + pitstop(tires,tires_type,engine,suspension,brakes,fuel) + change_power(1,0.9...0.5)

    def _extract_state(self, telemetry, scoring):
           
            return np.array([
                scoring["mLastLapTime"],
                scoring["mBestLapTime"],
                scoring["mCurrLapTime"],
                scoring["mInPits"],
                scoring["mNumPitstops"],
                scoring["mRaining"],
                scoring["mAmbientTemp"],
                scoring["mTrackTemp"],
                scoring["mEndET"],
                scoring["mCurrentET"],

                
        ], dtype=np.float32)

    def reset(self):
       
        self.state = [
            self.car.engine_health,
            self.car.suspension_health,
            self.car.brakes_health,
            self.car.fuel_level,
            self.car.tire_wear,
            self.car.tire_type_index,  # 0 - soft, 1 - medium, itd.
            self.car.power_setting,
            self.car.fuel_consumption,
            self.weather.surface_grip,
            self.weather.wet,
            self.weather.tire_wear,
            self.lap_time_start
        ]
        self.current_lap = 0
        self.current_state = 0

        self.lap_time = 0
    


    

    def step(self,action):
        possible_power_settings = [0.5,0.6,0.7,0.8,0.9,1]


        self.car.power_setting = possible_power_settings[action.nvec[8]]

        self.lap_time = self.fc_lap_time(action)



        self.state = [
            self.car.engine_health,
            self.car.suspension_health,
            self.car.brakes_health,
            self.car.fuel_level,
            self.car.tire_wear,
            self.car.tire_type_index,  # 0 - soft, 1 - medium, itd.
            self.car.power_setting,
            self.car.fuel_consumption,
            self.weather.surface_grip,
            self.weather.wet,
            self.weather.tire_wear,
            self.lap_time_start
        ]