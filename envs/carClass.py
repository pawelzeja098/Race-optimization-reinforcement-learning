class Car():
    def __init__(self):
        self.engine_health = 1
        self.suspension_health = 1
        self.brakes_health = 1
        self.fuel_level = 35
        self.tire_wear = 1
        self.tire_type_index = 0  # 0 - soft, 1 - medium, itd.
        self.power_setting = 0.8
        self.fuel_consumption = 3
        self.fuel_tank_capacity= 35
        self.engine_degrad = 0.02
        self.suspension_degrad = 0.01
        self.brakes_degrad = 0.04

          # Telemetria – nowa sekcja
        self.speed = 0.0
        self.speed_avg = 0.0
        self.speed_max = 0.0
        self.rpm = 0
        self.gear = 1
        self.throttle = 0.0
        self.brake = 0.0
        self.temp_engine = 90.0
        self.temp_oil = 90.0
        self.temp_gearbox = 80.0
        self.fuel_pressure = 4.0
        self.stint_time = 0.0
        self.lap_count = 0

        # Opony
        self.tire_temp = {"FL": (80.0, 85.0, 82.0), "FR": (80.0, 85.0, 82.0),
                          "RL": (80.0, 85.0, 82.0), "RR": (80.0, 85.0, 82.0)}
        self.tire_pressure = {"FL": 2.0, "FR": 2.0, "RL": 2.0, "RR": 2.0}
        self.tire_grip = {"FL": 0.95, "FR": 0.95, "RL": 0.95, "RR": 0.95}
        self.suspension_travel = {"FL": 25.0, "FR": 25.0, "RL": 25.0, "RR": 25.0}

        # Warunki zewnętrzne
        self.ambient_temp = 25.0
        self.track_temp = 35.0
        self.humidity = 50.0
        self.rain_level = 0.0
        self.surface_state = "dry"

        # Czujniki
        self.gps_position = (0.0, 0.0)
        self.angular_velocity = 0.0
        self.ers_energy = 0.0
        self.errors = {
            "engine_overheat": False,
            "sensor_fail": False
        }

