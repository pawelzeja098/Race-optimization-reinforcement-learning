import json

class Tire:
    def __init__(self, name, grip, durability, degradation_rate, wet_performance):
        self.name = name
        self.grip = grip
        self.durability = durability
        self.degradation_rate = degradation_rate
        self.wet_performance = wet_performance

    @staticmethod
    def load_from_file(filename):
        """
        Wczytuje dane o oponach z pliku JSON.
        """
        with open(filename, 'r') as file:
            data = json.load(file)
            tires = []
            for item in data:
                tire = Tire(
                    name=item["name"],
                    grip=item["grip"],
                    durability=item["durability"],
                    degradation_rate=item["degradation_rate"],
                    wet_performance=item["wet_performance"]
                )
                tires.append(tire)
            return tires
