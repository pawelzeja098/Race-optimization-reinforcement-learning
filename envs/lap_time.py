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
