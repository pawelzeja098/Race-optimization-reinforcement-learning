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
