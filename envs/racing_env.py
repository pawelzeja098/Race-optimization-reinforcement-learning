import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import csv
matplotlib.use('TkAgg')
import random
import json
import psutil
import os
import time
import threading
from ..core.carClasses.car_failure import CarFailure
from ..core.carClasses.race_car import RaceCar
from ..core.weatherClass.weather_class import Weather
from ..core.carClasses.tires_class import Tire
from ..core.failureClasses.brakes_failure_class import BrakesFailure
from ..core.failureClasses.engine_failure_class import EngineFailure
from ..core.failureClasses.suspension_failure_class import SuspensionFailure
from ..core.failureClasses.tires_failure_class import TiresFailure
from multiprocessing import Pool
import gymnasium as gym
from gymnasium import spaces
from ..env.carClass import Car

#pobranie możliwych usterek dla danego podzespołu i w jakim momencie zużycia jest ryzyko ich wystąpienia
failures_engine = EngineFailure.load_from_file()
thresholds_engine = [[failure.name, failure.engine_threshold ] for failure in failures_engine]

failures_brakes = BrakesFailure.load_from_file()
thresholds_brakes = [[failure.name, failure.brake_threshold ] for failure in failures_brakes]

failures_suspension = SuspensionFailure.load_from_file()
thresholds_suspension = [[failure.name, failure.suspension_threshold ] for failure in failures_suspension]

failures_tires = TiresFailure.load_from_file()
thresholds_tires = [[failure.name, failure.tires_threshold ] for failure in failures_tires]

failure_list = './data/failure_list.json'
weather_list = 'data/weather_conditions.json'
tire_list = 'data/tires_characteristics.json'




#Race enviroment class
class RacingEnv(gym.Env):
    def __init__(self):
        super(RacingEnv,self).__init__()

        race_list = ['data/race_simulation.json','data/race_simulation1.json','data/race_simulation2.json']

        with open(race_list[1], "r") as file:
            race_data = json.load(file)

        lap1 = race_data[0]
        self.weather = lap1["lap_data"]['weather']
        self.lap_time_start = lap1["lap_data"]["lap_time"]
        self.total_time = 0


        self.car = Car()
        
        #degradacja części
        
        
        
        self.weather = self.get_weather_by_name(self.weather,Weather.load_from_file(weather_list))

        self.tires_obj = Tire.load_from_file(tire_list)
        
        self.failures_list = []

        #tires - 0 - soft, 1 - medium, 2 - hard , 3 - wet 
        # self.state = [1, 1, 1, 1, 1, 0, 0.8] #Engine, suspension, brakes,fuel, tires_wear tires_type, car_power
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
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0.5,0,0,0,0,0]),
                                    high=np.array([1, 1, 1, 1, 1, 3, 1,20,1,1,1,1000]),
                                    dtype=np.float32)
        self.current_lap = 0
        self.current_state = 0

        self.lap_time = 0



        self.action_space = gym.spaces.MultiDiscrete([2,2,2,4,2,2,2,2,5]) #nothing + pitstop(tires,tires_type,engine,suspension,brakes,fuel) + change_power(1,0.9...0.5)


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

    def tire_degradation(self,tire_type_index,car_power):

        tires_degrad = self.get_tire_by_index(tire_type_index)
        tires_degrad = tires_degrad.degradation_rate + tires_degrad.degradation_rate * car_power * 0.8

        return tires_degrad
    
    def fuel_consumption(self):
        pass

    

    def fc_lap_time(self,action):
        
        engine_health = self.state[0]
        suspension_health = self.state[1]
        brakes_health = self.state[2]
        fuel_level = self.state[3]
        tire_wear = self.state[4]
        tire_type_index = self.state[5] # 0 - soft, 1 - medium, itd.
        car_power = self.state[6]
        fuel_consumption = self.state[7]
        surface_grip = self.state[8]
        wet = self.state[9]
        tire_wear_weather = self.state[10]  
        
        
        # tires_degrad = self.get_tire_by_index(tire_type_index)
        # tires_degrad = tires_degrad.degradation_rate + tires_degrad.degradation_rate * car_power * 0.8

        tires_degrad = self.tire_degradation(tire_type_index,car_power)

        lap_time_start = self.lap_time_start + self.lap_time_start * (1 - car_power) * 1.1

        
        #pobieranie danych pogody i usterek w danym okrążeniu
        # weather = lap["lap_data"]["weather"]
        failure = self.failure_generator(tire_wear,engine_health,suspension_health,brakes_health)
        
        #pobreanie obiektu z nazwy
        # failure = get_failure_by_name(failure,CarFailure.load_from_file(failure_list))
        if failure:
            self.failures_list.append(failure)
        

        lap_number = self.current_lap


        #obliczenie czasu okrążenia
        lap_time = self.lap_time_with_actuall_conditions(self.failures_list,lap_time_start,surface_grip,wet,engine_health,suspension_health,brakes_health,tire_wear,tire_type_index)
        fix_engine = action.nvec[4]
        fix_brakes = action.nvec[5]
        fix_suspension = action.nvec[6]
        tire_change = action.nvec[2]
        new_tires = action.nvec[3]
        fuel = action.nvec[7]
        pit_stop = False
        
        if action.nvec[1] == 1:

            pit_stop = True


        if pit_stop:
            tire_wear,failures_list, pitstop_time,fuel_level = self.pitstop(self.car,fuel_level, failures_list,fix_engine,fix_suspension, fix_brakes,tire_change,fuel ,new_tires)
            if tire_change == 1:
                tires_degrad = self.get_tire_by_index(new_tires)
                tires_degrad = tires_degrad.degradation_rate + tires_degrad.degradation_rate * car_power * 0.8

        else:
            pitstop_time = 0
        
        self.total_time += lap_time + pitstop_time
        tire_wear -= tires_degrad

        fuel_level -= fuel_consumption * car_power

        self.car.fuel_consumption = fuel_consumption * car_power
        
        # parts_wear = {part: parts_wear[part] - self.parts_degrad.get(part, 0) for part in parts_wear}
        #aktualzicaja stanu części
        self.car.engine_health -= self.car.engine_degrad
        self.car.suspension_health -= self.car.suspension_degrad
        self.car.brakes_health -= self.car.brakes_degrad
        

        return lap_time

    def lap_time_with_actuall_conditions(self,actuall_failures,lap_time,surface_grip,wet,engine_health,suspension_health,brakes_health,tire_wear,tire_type_index):
        #zwiększenie czasu okrążenia ze względu na usterki
        tires = self.get_tire_by_index(tire_type_index)
        
        try:

            reductions = [failure.speed_reduction for failure in actuall_failures]
        except:
            reductions = None
        max_red = 0
        if reductions:
            max_red = max(reductions)

        # weather = get_weather_by_name(weather,Weather.load_from_file(weather_list))
        wet_level = wet
        # tires = get_tire_by_name(tires,Tire.load_from_file(tire_list))

        tires_wet_cap = tires.wet_performance
        wet_reduction = 0
        #procentowe zwiększenie czasu okrazenia w zwiazku z mokroscia jezdzni 
        if wet_level > 0:
            wet_reduction = wet_level - tires_wet_cap
            if wet_reduction < 0:
                wet_reduction = 0

        
        grip_tires = tires.grip
        grip_weather = surface_grip

        #zwiększenie czasu okrążenia ze względu na typ użytych opon przy aktualnej pogodzie
        grip_red = grip_weather - grip_tires
        if grip_red < 0:
            grip_red = 0

        parts_sum_wear = 1 - sum([engine_health,suspension_health,brakes_health])


        return lap_time + lap_time * max_red + lap_time * wet_reduction + lap_time * grip_red + lap_time * 0.5 * (1 - tire_wear) + lap_time * (parts_sum_wear / 3)

    def pitstop(self,fuel_level, failures_list,fix_engine,fix_suspension, fix_brakes,tire_change,fuel ,new_tires):
        """
        Funkcja symulująca pitstop: wymiana opon, naprawa usterek, uzupełnienie paliwa.
        """
        total_repair_time = 0
        tires_change_time = 0
        t_to_garage = 0
        time_eng = 0
        time_brak = 0
        time_sus = 0
        time_refuel = 0
        actuall_failures = failures_list
        if tire_change:
        # Wymiana opon na nowe (przywracamy pełną wydajność)
            
            tires_list = Tire.load_from_file(tire_list)
            
            new_tire = new_tires
            self.car.tire_type_index = new_tire
            new_tire = self.get_tire_by_index(new_tire)
            tires = new_tire
            

            tires_wear = 1
            tires_change_time = 30
        
        # Uzupełnienie paliwa
        if fuel:
            refuel = self.car.fuel_tank_capacity - self.car.fuel_level  # Maksymalna pojemność baku
            time_refuel = 10 + refuel * 2 #czas spędzony na dolanie x litrów paliwa
            self.car.fuel_level = self.car.fuel_tank_capacity



        if fix_engine == 1:
        
            time_eng = 300
            self.car.engine_health = 1
        
        if fix_brakes == 1:
            
            time_brak = 340
            self.car.brakes_health = 1
        if fix_suspension == 1:
            
            time_sus = 460
            self.car.suspension_health = 1
        failure_names = []
        if actuall_failures:
        # Naprawa usterek
        
            for failure in actuall_failures:
                
                failure_names.append(failure[0].name)
                total_repair_time += failure[0].fixtime  # Sumujemy czas naprawy

        # Można zresetować awarie po naprawach
            actuall_failures.clear()
            t_to_garage = 60
        
        # Pitstop trwa również określony czas
        pitstop_time = 120 + total_repair_time + time_refuel + tires_change_time + t_to_garage + time_sus + time_brak + time_eng # Zliczamy czas pitstopu i naprawy
        
        # pitstop_data = {
        #     "repairs" : failure_names,
        #     "fuel_amount" : refuel,
        #     "tires_change" : tire_change,
        #     "time in pitstop" : pitstop_time

        # }
        
        # Zwracamy nowe opony, czas pitstopu oraz naprawy
        try:
            return tires.name,tires_wear,actuall_failures, pitstop_time,fuel_level
        except:
            return tires,tires_wear,actuall_failures, pitstop_time,fuel_level


    def failure_generator(self,tire_wear,engine_health,suspension_health,brakes_health):
        tire_failure = False
        possible_failures_breaks = []
        possible_failures_engine = []
        possible_failures_suspension = []
        engine_wear = engine_health
        suspension_wear = suspension_health
        brakes_wear = brakes_health

        lst_fail_b = []
        lst_fail_e = []
        lst_fail_s = []

        random_failure = []

        for threshold in thresholds_brakes:
            if threshold[1] > brakes_wear:
                possible_failures_breaks.append(threshold[0])
        
        for threshold in thresholds_engine:
            if threshold[1] > engine_wear:
                possible_failures_engine.append(threshold[0])
        
        for threshold in thresholds_suspension:
            if threshold[1] > suspension_wear:
                possible_failures_suspension.append(threshold[0])
        
        if possible_failures_breaks:
            for name in possible_failures_breaks:
                lst_fail_b.append(self.get_failure_brakes_by_name(name)) 
                random_failure.append(self.choose_random_failure(lst_fail_b,brakes_wear))

        if possible_failures_engine:
            for name in possible_failures_engine:
                lst_fail_e.append(self.get_failure_engine_by_name(name)) 
                random_failure.append(self.choose_random_failure(lst_fail_e,engine_wear))
        
        if possible_failures_suspension:
            for name in possible_failures_suspension:
                lst_fail_s.append(self.get_failure_suspension_by_name(name))
                random_failure.append(self.choose_random_failure(lst_fail_s,suspension_wear))

        
        if self.car.tire_wear < 0.65:
            tire_fail_prob = (1 - tire_wear)**2
            tire_failure = random.choices([True, False], weights=[tire_fail_prob, 1 - tire_fail_prob], k=1)[0]

        if tire_failure:
            if not random_failure:
                random_failure = []
            tire_fail = random.choices(["Tire puncture","Tire blowout"], weights=[tire_wear, 1 - tire_wear], k=1)[0]
            tire_fail = self.get_failure_tires_by_name(tire_fail)
            random_failure.append(tire_fail)


        
        return random_failure

    def choose_random_failure(self,failures,part_wear):
        
        probabilities = [failure.propability for failure in failures]
        weights = np.array(probabilities) * (1 - part_wear)
        chosen_failure = random.choices(failures, weights, k=1)[0]
        
        return chosen_failure

    def get_tire_by_index(self,idx):
        tires = ["Soft","Medium","Hard","Wet"]
        return self.get_tire_by_name(tires[idx])

    def get_tire_by_name(self,name, tires  = None):
        tires = self.tires_obj
        for tire in tires:
            if tire.name.lower() == name.lower():  # Ignoruje wielkość liter
                return tire
        return None  # Jeśli nie znaleziono opony
    

    # def get_failure_by_name(name,failure_list):
    #     for failure in failure_list:
    #         if  failure.name.lower() == name.lower():  # Ignoruje wielkość liter
    #             return failure
    #     return None

    def get_failure_engine_by_name(self,name,failure_list=failures_engine):
        for failure in failure_list:
            if  failure.name.lower() == name.lower():  # Ignoruje wielkość liter
                return failure
        return None

    def get_failure_suspension_by_name(self,name,failure_list=failures_suspension):
        for failure in failure_list:
            if  failure.name.lower() == name.lower():  # Ignoruje wielkość liter
                return failure
        return None

    def get_failure_tires_by_name(self,name,failure_list=failures_tires):
        for failure in failure_list:
            if  failure.name.lower() == name.lower():  # Ignoruje wielkość liter
                return failure
        return None

    def get_failure_brakes_by_name(self,name,failure_list=failures_brakes):
        for failure in failure_list:
            if  failure.name.lower() == name.lower():  # Ignoruje wielkość liter
                return failure
        return None

    def get_weather_by_name(self,name,weather_list):
        for weather in weather_list:
            if  weather.name.lower() == name.lower():  # Ignoruje wielkość liter
                return weather
        return None  
    


# car = RaceCar(
#         make="Toyota",
#         model="GR010 Hybrid",
#         top_speed=340,
#         horsepower=680,
#         weight=1040,
#         fuel_tank_capacity=35,
#         average_fuel_consumption=3, #power*avg_fuel_consumption``
#         lap_time=350
#         )

 



env = RacingEnv()












# if __name__ == "__main__":
#     start_memory = monitor_memory()
#     start_time = time.perf_counter()
#     abc_algorithm_demo(10, 10, 10, 1)
#     stop_time = time.perf_counter()
#     end_memory = monitor_memory()

#     print(f"Memory used: {end_memory - start_memory:.2f} MB")
#     print(f"Execution time: {stop_time - start_time}")
