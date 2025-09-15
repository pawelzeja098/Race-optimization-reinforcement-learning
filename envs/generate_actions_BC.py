
import json
import numpy as np

def generate_actions_BC(data_race_scoring = "data/scoring_data.json", data_race_telemetry = "data/telemetry_data.json"):
    with open(data_race_scoring, "r") as file:
        scoring_data = json.load(file)
    with open(data_race_telemetry, "r") as file:
        telemetry_data = json.load(file)
    
    actions = []

    for i in range(len(telemetry_data) - 4):  # -4 bo patrzymy na następny pitstop
        telemetry = telemetry_data[i]
        scoring = scoring_data[i]

        action = [0, 0, 0, 0, 0]  # [pitstop, confirm, tire_type, repair, fuel_units]
        if telemetry["mCurrentSector"] not in [0,1, 2]:
            if i != 0:
                prev_sector = telemetry_data[i-1]["mCurrentSector"]
                # zwiększamy o 1 i bierzemy modulo 3, żeby wracało do 0 po 2
                telemetry["mCurrentSector"] = (prev_sector + 1) % 3
            else:
                # dla pierwszego elementu, jeśli nieprawidłowy, ustaw na 0
                telemetry["mCurrentSector"] = 0

        # --- sprawdzamy decyzję w ostatnim sektorze
        if telemetry["mCurrentSector"] == 0:
            next_scoring = scoring_data[i+3]
            next_telemetry = telemetry_data[i+3]

            next_telemetry_after_pit = telemetry_data[i+4] # stan po pitstopie (jeśli był)

            if next_scoring["mInPits"]:
                action[0] = 1  # pitstop TAK

                # fuel_units: ile zatankowano -> różnica między fuel_capacity a aktualnym fuel
                fuel_after = next_telemetry_after_pit["mFuel"]
                fuel_before_pit= next_telemetry["mFuel"]
                fuel_units = int(round((fuel_after - fuel_before_pit) // 5.0))
                action[3] = max(0, min(fuel_units, 22))  # clamp do 23 akcji

                # tire type
                action[1] = next_telemetry_after_pit["mFrontTireCompoundIndex"]

                # repair (np. sprawdź dentSeverity: jeśli spadło, to naprawa)
                if sum(next_telemetry_after_pit["mDentSeverity"]) < sum(telemetry["mDentSeverity"]):
                    action[2] = 1

            actions.append(action)

        elif telemetry["mCurrentSector"] == 2:
            
            next_scoring = scoring_data[i+1]
            next_telemetry = telemetry_data[i+1]

            next_telemetry_after_pit = telemetry_data[i+2] # stan po pitstopie (jeśli był)

            if next_scoring["mInPits"]:
                if action[0] == 1:
                    action[1] = 1  # confirm pitstop
                    actions.append(action)
                    continue
                else:
                    action[1] = 0  # change decision
                action[0] = 1  # pitstop TAK

                # fuel_units: ile zatankowano -> różnica między fuel_capacity a aktualnym fuel
                fuel_after = next_telemetry_after_pit["mFuel"]
                fuel_before_pit= next_telemetry["mFuel"]
                fuel_units = int(round((fuel_after - fuel_before_pit) // 5.0))
                action[3] = max(0, min(fuel_units, 22))  # clamp do 23 akcji

                # tire type
                action[1] = next_telemetry_after_pit["mFrontTireCompoundIndex"]

                # repair (np. sprawdź dentSeverity: jeśli spadło, to naprawa)
                if sum(next_telemetry_after_pit["mDentSeverity"]) < sum(telemetry["mDentSeverity"]):
                    action[2] = 1
                actions.append(action)
        else:
            continue

            

    # zapisz do json
    return actions

    

            
# generate_actions_BC()


    