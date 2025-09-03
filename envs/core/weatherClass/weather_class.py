import json
class Weather:
    def __init__(self,name,failure_probability,speed_reduction,speed_reduction_night,surface_grip,tire_wear,next_weather,wet) -> None:
        self.name = name
        self.failure_probability = failure_probability
        self.speed_reduction = speed_reduction
        self.speed_reduction_night = speed_reduction_night
        self.surface_grip = surface_grip
        self.tire_wear = tire_wear
        self.next_weather = next_weather
        self.wet = wet


    @staticmethod
    def load_from_file(filename):
        with open(filename, 'r') as file:
            data = json.load(file)
            weathers = []
            for item in data:
                weather = Weather(
                    name=item["name"],
                    failure_probability = item["failure_probability"],
                    speed_reduction=item["speed_reduction"],
                    speed_reduction_night=item["speed_reduction_night"],
                    surface_grip = item["surface_grip"],
                    tire_wear = item["tire_wear"],
                    next_weather = item["next_weather"],
                    wet = item["wet"]
                )
                weathers.append(weather)
            return weathers