import json
import random

class SuspensionFailure:
    name: str
    fixtime: int  # sec
    garage: bool  # czy można naprawić w garażu
    stock_number: int  # ile razy można ponownie naprawić daną usterkę
    propability: float  # prawdopodobieństwo wystąpienia (waga)
    speed_reduction: float  # procent vmax
    speed_reduction_night: float  # procent vmax w nocy
    failure_deterioration: float  # pogarszanie się usterki per okrążenie
    next_failure: str  # następna usterka, gdy nie naprawiona
    fuel_penalty: float  # zwiększone zużycie paliwa 1l/100km
    chance_of_repair_failure: float  # szansa na nieudaną naprawę
    suspension_threshold: float  # próg zużycia zawieszenia, przy którym występuje awaria
    

    def __init__(self, name, fixtime, garage, stock_number, propability, speed_reduction, speed_reduction_night, failure_deterioration, next_failure, fuel_penalty, chance_of_repair_failure, suspension_threshold ) -> None:
        self.name = name
        self.fixtime = fixtime
        self.garage = garage
        self.stock_number = stock_number
        self.propability = propability
        self.speed_reduction = speed_reduction
        self.speed_reduction_night = speed_reduction_night
        self.failure_deterioration = failure_deterioration
        self.next_failure = next_failure
        self.fuel_penalty = fuel_penalty
        self.chance_of_repair_failure = chance_of_repair_failure
        self.suspension_threshold = suspension_threshold
        

    @staticmethod
    def load_from_file(filename = "./data/suspension_failures.json"):
        with open(filename, 'r') as file:
            data = json.load(file)
            failures = []
            for item in data:
                failure = SuspensionFailure(
                    name=item["name"],
                    fixtime=item["fixtime"],
                    garage=item["garage"] == "True",
                    stock_number=item["stock_number"],
                    propability=item["propability"],
                    speed_reduction=item["speed_reduction"],
                    speed_reduction_night=item["speed_reduction_night"],
                    failure_deterioration=item["failure_deterioration"],
                    next_failure=item["next_failure"],
                    fuel_penalty=item["fuel_penalty"],
                    chance_of_repair_failure=item["chance_of_repair_failure"],
                    suspension_threshold=item["suspension_threshold"]
                )
                failures.append(failure)
            return failures