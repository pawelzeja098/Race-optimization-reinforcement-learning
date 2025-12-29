# scoring_file_raw = "data/raw_races/scoring_data.json"
# telem_file_raw = "data/raw_races/telemetry_data.json"
import numpy as np
import sqlite3
import json
import copy
import matplotlib.pyplot as plt
# from generate_actions_BC import generate_actions_BC
# from generate_state_BC import extract_state

# scoring_file_filtered = "data/filtered_races/scoring_data.json"
# telem_file_filtered = "data/filtered_races/telemetry_data.json"


def filtr_json_files(telem_file_raw, scoring_file_raw):
    print("Filtracja plików:", telem_file_raw, "i", scoring_file_raw)
    data_telem = json.load(open(telem_file_raw ))
    data_scoring = json.load(open(scoring_file_raw))
    raw_data_telemetry = []
    raw_data_scoring = []

    filtered_data_scoring = []
    filtered_data_telemetry = []
    for entry in data_telem:
        if entry['data'].get("Type") == "TelemInfoV01":
            raw_data_telemetry.append(entry['data'])
    for entry in data_scoring:
        if entry['data'].get("Type") == "ScoringInfoV01":
            raw_data_scoring.append(entry['data'])

    # with open("data/telemetry_data_filtered.json", "w") as f:
    #     json.dump(filtered_data, f, indent=2)
    scoring_records_len = 0
    data_before_start = 0
    start_flag = False
    # data_saved_scoring = 0
    start_lights_flag = False
    lap_dist_hist = []

    for entry in raw_data_scoring:
        # for vehicle in raw_data_scoring.get("mVehicles", []):
        start = entry["mStartLight"]
        
        if start == 6:
            start_lights_flag = True
        vehicle = entry.get("mVehicles")
        time_start = vehicle[0]["mTimeIntoLap"]
        if (time_start < 0 or not start_lights_flag) and not start_flag:
            data_before_start += 1

            
            continue
        start_lights_flag = False
        start_flag = True
        # if entry.get("mVehicles")["mFinishStatus"] == 1:
        #     break
        
        # data_saved_scoring += 1
        #     if vehicle.get("mIsPlayer"):
        vehicle = entry.get("mVehicles")

        wanted_weather_keys = ["mRaining","mAmbientTemp","mTrackTemp","mEndET", "mCurrentET","mAvgPathWetness"]
        subset_weather = {k: entry.get(k) for k in wanted_weather_keys}
        subset_weather["mTotalLapDistance"] = entry["mLapDist"]
        
        
        vehicle_sector = vehicle[0]["mSector"]
        
        curr_sector = vehicle_sector
        wanted_keys = ["mLastLapTime","mBestLapTime","mCurrLapTime","mNumPitstops","mNumPenalties","mInPits","mFinishStatus","mLapDist","mSector","mTotalLaps"]
        if vehicle[0]["mFinishStatus"] == 1:
            subset_scoring_vehicle = {k: vehicle[0].get(k) for k in wanted_keys}

        
        # print(json.dumps(vehicle, indent=2))
        # scoring_records.append(data)
            filtered_data_scoring.append({**subset_scoring_vehicle, **subset_weather})
            scoring_records_len += 1

            break

        subset_scoring_vehicle = {k: vehicle[0].get(k) for k in wanted_keys}
        if subset_scoring_vehicle["mLapDist"] < 20:
            lap_dist_hist.append(subset_scoring_vehicle["mLapDist"])
        if subset_scoring_vehicle["mLapDist"] < 0:
            print("Negative lap distance detected at record", scoring_records_len, "value:", subset_scoring_vehicle["mLapDist"])
        # print(json.dumps(vehicle, indent=2))
        # scoring_records.append(data)
        filtered_data_scoring.append({**subset_scoring_vehicle, **subset_weather})
        scoring_records_len += 1
    
    # plt.plot(lap_dist_hist)
    # plt.title("Lap distance over time")
    # plt.show()

    
    telemetry_records_len = 0
    count_before_start = 0
    data_saved_telemetry = 0
    curr_tel = 0
    delta_fuel = 0.0
    delta_tires = [0.0,0.0,0.0,0.0]
    delta_fuel_hist = []
    delta_tires_hist = []
    for entry in raw_data_telemetry:
        refueled_amount = 0.0
        changed_tires_flag = False
        refueled_flag = False

        if count_before_start < data_before_start:
            count_before_start += 1
            curr_tel += 1
            continue
        if telemetry_records_len >= scoring_records_len:
            break
        # if data_saved_telemetry >= data_saved_scoring:
        #     break
        # data_saved_telemetry += 1
        wanted_keys = ["mFuel", "mFuelCapacity","mWheel","mDentSeverity","mFrontTireCompoundIndex","mCurrentSector","mLapNumber","mLastImpactET","mLastImpactMagnitude","multiplier","is_repairing"]
        vehicle = raw_data_scoring[curr_tel].get("mVehicles")

        if curr_tel > 0:
            for i in range(4):
                delta_tires[i] = raw_data_telemetry[curr_tel-1]["mWheel"][i]["mWear"] - entry["mWheel"][i]["mWear"] 
                if delta_tires[i] < 0:
                    delta_tires[i] = 0.0
            # print("Delta opon w stepie", telemetry_records_len, ":", delta_tires)
            # if delta_tires[0] < 0:
            #     print("Negative delta tires[0] detected at step", telemetry_records_len, "value:", delta_tires[0], "max steps:", scoring_records_len)

            delta_fuel = raw_data_telemetry[curr_tel-1]["mFuel"] - entry["mFuel"]
            
            
        else:
            delta_tires = [0.0,0.0,0.0,0.0]
        
        # if vehicle[0]["mInPits"] is True:
        #     # print("Distance in lap during pitstop:", vehicle[0]["mLapDist"], "at telemetry record:", telemetry_records_len)

        

        if entry["mWheel"][0]["mWear"] > raw_data_telemetry[curr_tel-1]["mWheel"][0]["mWear"] and curr_tel > 0:
            vehicle = raw_data_scoring[curr_tel].get("mVehicles")
            delta_tires = [0.0,0.0,0.0,0.0]
            if vehicle[0]["mInPits"] is True:
                # print("Zmiana opon 0 wykryta, w stepie:", telemetry_records_len)
                # print(entry["mWheel"][0]["mWear"])
                skonczone_w_step = telemetry_records_len

                changed_tires_flag = True
      
        FUEL_THRESHOLD = 0.01
        if entry["mFuel"] > raw_data_telemetry[curr_tel-1]["mFuel"] + FUEL_THRESHOLD and curr_tel > 0:
            vehicle = raw_data_scoring[curr_tel].get("mVehicles")

            delta_fuel = 0.0
            
            # print(f"Fuel level: {entry['mFuel']} at ET {vehicle[0]['mTotalLaps']}, {telemetry_records_len} record")
            if vehicle[0]["mInPits"] is True:
                # refueled_flag = True
                refueled_amount = entry["mFuel"] - raw_data_telemetry[curr_tel-1]["mFuel"]
                # print(f"Refueled {round(refueled_amount,5)} liters at ET {vehicle[0]['mTotalLaps']}, {telemetry_records_len} record")
                skonczone_w_step = telemetry_records_len
                # print(f"Refueled at ET {vehicle[0]['mTotalLaps']}, {telemetry_records_len} record")
            # changed_tires_flag = True
        if entry["mDentSeverity"] != raw_data_telemetry[curr_tel-1]["mDentSeverity"] and curr_tel > 0:
            vehicle = raw_data_scoring[curr_tel].get("mVehicles")
            if vehicle[0]["mInPits"] is True:
                print("Naprawa uszkodzen wykryta, w stepie:", telemetry_records_len , "Wczesniejszze uszkodzenia:", raw_data_telemetry[curr_tel-1]["mDentSeverity"], "Aktualne uszkodzenia:", entry["mDentSeverity"])
                print("zajelo to stepow:", telemetry_records_len - skonczone_w_step, "ile uszkodzen naprawiono:", sum(raw_data_telemetry[curr_tel-1]["mDentSeverity"])-sum(entry["mDentSeverity"]) )
                steps_duration = telemetry_records_len - skonczone_w_step
                for i in range(8):

                    if entry["mDentSeverity"][i] < raw_data_telemetry[curr_tel-1]["mDentSeverity"][i]:
                        print(f"Element {i} naprawiony o wartosc {raw_data_telemetry[curr_tel-1]['mDentSeverity'][i] - entry['mDentSeverity'][i]}")
                
                start_index = len(filtered_data_telemetry) - steps_duration
                end_index = len(filtered_data_telemetry)

                for i in range(start_index, end_index):
                    if i >= 0: # Zabezpieczenie
                        filtered_data_telemetry[i]["is_repairing"] = 1
               # print(entry["mDentSeverity"])
                # print(raw_data_telemetry[curr_tel-1]["mDentSeverity"])
        avg_temp = 0
        
        subset = {k: entry.get(k) for k in wanted_keys}
        subset["delta_tires"] = delta_tires
        subset["delta_fuel"] = delta_fuel
        subset["refueled_amount"] = refueled_amount
        subset["is_repairing"] = 0
        subset["changed_tires_flag"] = int(changed_tires_flag)
        delta_tires_hist.append(delta_tires[0])
        delta_fuel_hist.append(delta_fuel)
        if delta_fuel < 0:
                
                print("Negative delta fuel detected at step", telemetry_records_len, "value:", delta_fuel, "max steps:", scoring_records_len)
        # subset["refueled_flag"] = int(refueled_flag)
      
        filtered_data_telemetry.append(subset)
        telemetry_records_len += 1
        curr_tel += 1
        

    return filtered_data_telemetry, filtered_data_scoring



def extract_state(telem_file_raw, scoring_file_raw):
        filtered_data_telemetry, filtered_data_scoring = filtr_json_files(telem_file_raw,scoring_file_raw)
       
        data_state = []
        
        scoring_all = filtered_data_scoring
        telemetry_all = filtered_data_telemetry
        len_data = len(telemetry_all)
        endET = -1
        print("Kolejny wyscig")
        for i in range(len(telemetry_all)):

            telemetry = copy.deepcopy(telemetry_all[i])
            scoring = copy.deepcopy(scoring_all[i])
            #SPRAWDZIC CZY NA POCZATKU JEST 0
          

        
            if telemetry["mLastImpactET"] <= 0:
                telemetry["mLastImpactET"] = 0.0
            else:
                telemetry["mLastImpactET"] = scoring["mCurrentET"] - telemetry["mLastImpactET"]

                if telemetry["mLastImpactET"] < 0:
                    telemetry["mLastImpactET"] = 0.0
            
            last_lap = scoring["mLastLapTime"]
            best_lap = scoring["mBestLapTime"]
            has_last_lap = 1.0 if last_lap > 0 else 0.0
            if last_lap <= 0.0:
                last_lap = 0  # placeholder
                best_lap = 0
            
            # print(telemetry["mLastImpactMagnitude"])
            # print(telemetry_all[i-1]["mLastImpactMagnitude"] if i > 0 else "No previous data")

            if i > 0 and telemetry["mLastImpactMagnitude"] == telemetry_all[i-1]["mLastImpactMagnitude"]:
                telemetry["mLastImpactMagnitude"] = 0.0
                impact_flag = 0.0
            elif i == 0:
                impact_flag = 0.0
                telemetry["mLastImpactMagnitude"] = 0.0
            else:
                impact_flag = 1.0
                # print("Impact", telemetry["mLastImpactMagnitude"])
            
            if scoring["mEndET"] < 0:
                endET = scoring_all[i+4]["mEndET"]

            else:
                endET = scoring["mEndET"]

            corrected_dist_meters = scoring["mLapDist"] % scoring["mTotalLapDistance"]

            lap_dist_sin = np.sin(2 * np.pi * (corrected_dist_meters / scoring["mTotalLapDistance"]))

            lap_dist_cos = np.cos(2 * np.pi * (corrected_dist_meters / scoring["mTotalLapDistance"]))
            curr_step = i/len_data
        #     data_state_per = [
        #         #dane do przewidzenia
        #         #dane ciągłe
                
        #         # scoring["mCurrLapTime"],
                
        #         # round(scoring["mLapDist"]/scoring["mTotalLapDistance"],5),
        #         lap_dist_sin,
        #         lap_dist_cos,
        #         # round(scoring["mCurrentET"]/endET,5),
        #         sum(telemetry["mWheel"][0]["mTemperature"])/len(telemetry["mWheel"][0]["mTemperature"]),
        #         sum(telemetry["mWheel"][1]["mTemperature"])/len(telemetry["mWheel"][1]["mTemperature"]),
        #         sum(telemetry["mWheel"][2]["mTemperature"])/len(telemetry["mWheel"][2]["mTemperature"]),
        #         sum(telemetry["mWheel"][3]["mTemperature"])/len(telemetry["mWheel"][3]["mTemperature"]),

        #         telemetry["mFuel"]/telemetry["mFuelCapacity"],
        #         # telemetry["delta_tires"][0],
        #         # telemetry["delta_tires"][1],
        #         # telemetry["delta_tires"][2],
        #         # telemetry["delta_tires"][3],
                
                

        #         #dane pomocnicze ciągłe
        #         scoring["mAvgPathWetness"],
        #         telemetry['mWheel'][0]['mWear'],  
        #         telemetry["mWheel"][1]["mWear"],
        #         telemetry["mWheel"][2]["mWear"],
        #         telemetry["mWheel"][3]["mWear"],
        #         curr_step,
        #         telemetry["refueled_amount"]/telemetry["mFuelCapacity"],

        #         # telemetry["mLastImpactET"],
        #         telemetry["mLastImpactMagnitude"],
        #         scoring["mNumPenalties"],
        #         scoring["mRaining"],
        #         round(scoring["mAmbientTemp"], 2),
        #         round(scoring["mTrackTemp"], 2),
        #         round(endET,5),

        #         #dane dyskretne pomocniczne
                
        #         impact_flag,
        #         telemetry["mDentSeverity"][0],  # Not defined which part of the car this refers to each index
        #         telemetry["mDentSeverity"][1],
        #         telemetry["mDentSeverity"][2], 
        #         telemetry["mDentSeverity"][3],
        #         telemetry["mDentSeverity"][4],
        #         telemetry["mDentSeverity"][5],
        #         telemetry["mDentSeverity"][6], 
        #         telemetry["mDentSeverity"][7],

        #         # has_last_lap,
        #         scoring["mFinishStatus"],
        #         scoring["mTotalLaps"],
        #         scoring["mSector"],
        #         scoring["mNumPitstops"],
        #         int(scoring["mInPits"]),
        #         telemetry["mFrontTireCompoundIndex"],
        #         telemetry["multiplier"],
        #         telemetry["changed_tires_flag"],
        #         telemetry["is_repairing"],
        #         # telemetry["refueled_flag"],
        #         #DO PRZEWIDZENIA ale nie do X
        #         telemetry["delta_fuel"]/telemetry["mFuelCapacity"],
                
        #         telemetry["delta_tires"][0],
        #         telemetry["delta_tires"][1],
        #         telemetry["delta_tires"][2],
        #         telemetry["delta_tires"][3],

        #         #ciągłe nie używane do trenowania
        #         # last_lap,
        #         # best_lap,
                
                
        # ]
            data_state_per = [
                #NO SCALER
                lap_dist_sin,
                lap_dist_cos,
                telemetry["mFuel"]/telemetry["mFuelCapacity"],
                scoring["mAvgPathWetness"],
                telemetry['mWheel'][0]['mWear'],  
                telemetry["mWheel"][1]["mWear"],
                telemetry["mWheel"][2]["mWear"],
                telemetry["mWheel"][3]["mWear"],
                curr_step,
                telemetry["refueled_amount"],
                scoring["mRaining"],
                impact_flag,
                # scoring["mFinishStatus"],
                int(scoring["mInPits"]),
                telemetry["changed_tires_flag"],
                telemetry["is_repairing"],
                #MIN-MAX SCALER
                scoring["mNumPenalties"],
                telemetry["mDentSeverity"][0],  # Not defined which part of the car this refers to each index
                telemetry["mDentSeverity"][1],
                telemetry["mDentSeverity"][2], 
                telemetry["mDentSeverity"][3],
                telemetry["mDentSeverity"][4],
                telemetry["mDentSeverity"][5],
                telemetry["mDentSeverity"][6], 
                telemetry["mDentSeverity"][7],
                scoring["mTotalLaps"],
                scoring["mSector"],
                scoring["mNumPitstops"],
                telemetry["mFrontTireCompoundIndex"],
                telemetry["multiplier"],
                #ROUBST SCALER
                sum(telemetry["mWheel"][0]["mTemperature"])/len(telemetry["mWheel"][0]["mTemperature"]),
                sum(telemetry["mWheel"][1]["mTemperature"])/len(telemetry["mWheel"][1]["mTemperature"]),
                sum(telemetry["mWheel"][2]["mTemperature"])/len(telemetry["mWheel"][2]["mTemperature"]),
                sum(telemetry["mWheel"][3]["mTemperature"])/len(telemetry["mWheel"][3]["mTemperature"]),
                telemetry["mLastImpactMagnitude"],
                scoring["mAmbientTemp"],
                scoring["mTrackTemp"],
                round(endET,5),

                #DLA Y
                #NO SCALER
                lap_dist_sin,
                lap_dist_cos,
                #MIN-MAX SCALER
                telemetry["delta_fuel"],
                telemetry["delta_tires"][0],
                telemetry["delta_tires"][1],
                telemetry["delta_tires"][2],
                telemetry["delta_tires"][3],
                #ROUBST SCALER
                sum(telemetry["mWheel"][0]["mTemperature"])/len(telemetry["mWheel"][0]["mTemperature"]),
                sum(telemetry["mWheel"][1]["mTemperature"])/len(telemetry["mWheel"][1]["mTemperature"]),
                sum(telemetry["mWheel"][2]["mTemperature"])/len(telemetry["mWheel"][2]["mTemperature"]),
                sum(telemetry["mWheel"][3]["mTemperature"])/len(telemetry["mWheel"][3]["mTemperature"]),


            ]





            
            
    
          
            data_state.append(data_state_per)

       
        # with open("data/state_data.json", "w") as file:
        #     json.dump(data_state, file, indent=2)

        return data_state



def save_to_db_tests(telem_file_raw, scoring_file_raw):
    # przykładowe stany i akcje
    states_x = extract_state(telem_file_raw, scoring_file_raw)

    print(len(states_x))


    # zamieniamy listy na JSON
    states_json = json.dumps(states_x)
   
    # tworzymy bazę / tabelę
    conn = sqlite3.connect("data/db_states_for_regress/race_data_states_tests.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS races (
        race_id INTEGER PRIMARY KEY AUTOINCREMENT,
        states_json TEXT
        
    )
    """)

    # zapisujemy rekord, nie podajemy race_id, SQLite automatycznie nada
    cursor.execute("""
    INSERT INTO races (states_json)
    VALUES (?)
""", (states_json,)) 

    conn.commit()
    conn.close()




def load_from_db():
    conn = sqlite3.connect("data/db_states_for_regress/race_data_states.db")
    cursor = conn.cursor()

    # wybierz wszystkie rekordy z tabeli
    cursor.execute("SELECT * FROM races")
    rows = cursor.fetchall()

    for row in rows:
        race_id, states_json = row
        
        
        # zamiana JSON z powrotem na listę
        states = json.loads(states_json)
        
        
        print(f"Number of states: {len(states)}")
        print("Przykładowy state:", states)
       
        print("-"*40)

    conn.close()



def delete_from_db(race_id):
    conn = sqlite3.connect("data/db_states_for_regress/race_data_states.db")
    cursor = conn.cursor()

    race_id_to_delete = 1
    cursor.execute("DELETE FROM races WHERE race_id = ?", (race_id_to_delete,))
    conn.commit()
    conn.close()
