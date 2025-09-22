# scoring_file_raw = "data/raw_races/scoring_data.json"
# telem_file_raw = "data/raw_races/telemetry_data.json"
import numpy as np
import sqlite3
import json
import numpy as np
# from generate_actions_BC import generate_actions_BC
# from generate_state_BC import extract_state

# scoring_file_filtered = "data/filtered_races/scoring_data.json"
# telem_file_filtered = "data/filtered_races/telemetry_data.json"


def filtr_json_files(telem_file_raw, scoring_file_raw):
    data_telem = json.load(open(telem_file_raw ))
    data_scoring = json.load(open(scoring_file_raw))
    raw_data_telemetry = []
    raw_data_scoring = []

    filtered_data_scoring = []
    filtered_data_telemetry = []
    for entry in data_telem:
        if entry.get("Type") == "TelemInfoV01":
            raw_data_telemetry.append(entry)
    for entry in data_scoring:
        if entry.get("Type") == "ScoringInfoV01":
            raw_data_scoring.append(entry)

    # with open("data/telemetry_data_filtered.json", "w") as f:
    #     json.dump(filtered_data, f, indent=2)
    scoring_records_len = 0
    data_before_start = 0
    start_flag = False

    for entry in raw_data_scoring:
        # for vehicle in raw_data_scoring.get("mVehicles", []):
        start = entry["mStartLight"]
        if start == 0 and not start_flag:
            data_before_start += 1
            start_flag = True
            continue

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

        # print(json.dumps(vehicle, indent=2))
        # scoring_records.append(data)
        filtered_data_scoring.append({**subset_scoring_vehicle, **subset_weather})
        scoring_records_len += 1
    
    telemetry_records_len = 0
    count_before_start = 0
    for entry in raw_data_telemetry:
        if count_before_start < data_before_start:
            count_before_start += 1
            continue
        if telemetry_records_len >= scoring_records_len:
            break
        wanted_keys = ["mFuel", "mFuelCapacity","mWheel","mDentSeverity","mFrontTireCompoundIndex","mCurrentSector","mLapNumber","mLastImpactET","mLastImpactMagnitude","multiplier"]

        avg_temp = 0
        subset = {k: entry.get(k) for k in wanted_keys}
        # for wheel in subset["mWheel"]:
        #     for temp in wheel.get("mTemperature", 0):
        #         avg_temp += temp
        #     avg_temp /= len(wheel.get("mTemperature", []))

       #Mean values for wheels

        # wheels = entry.get("mWheel", [])
        # avg_wear = 0
        # avg_brake_temp = 0
        # avg_temp = 0
        # for wheel in wheels:
        #     avg_wear += wheel.get("mWear", 0)
        #     # avg_brake_temp += wheel.get("mBrakeTemp", 0)
        #     # avg_temp += wheel.get("mTemperature", 0)
        #     for temp in wheel.get("mTemperature", 0):
        #         avg_temp += temp
        #     avg_temp /= len(wheel.get("mTemperature", []))
        # avg_wear /= len(wheels)
        # # avg_brake_temp /= len(wheels)
        # avg_temp /= len(wheels)

        # subset["mWheel"] = {
        #     "mWear": avg_wear,
        #     "mBrakeTemp": avg_brake_temp,
        #     "mTemperature": avg_temp,
        # }
        filtered_data_telemetry.append(subset)
        telemetry_records_len += 1
        # print(json.dumps(subset, indent=2))
    

    return filtered_data_telemetry, filtered_data_scoring
    # with open(telem_file_filtered, "w") as f:
    #     json.dump(filtered_data_telemetry, f, indent=2)
    # with open(scoring_file_filtered, "w") as f:
    #     json.dump(filtered_data_scoring, f, indent=2)


def extract_state(telem_file_raw, scoring_file_raw):
        filtered_data_telemetry, filtered_data_scoring = filtr_json_files(telem_file_raw,scoring_file_raw)
        
        data_state = []
        
        scoring_all = filtered_data_scoring
        telemetry_all = filtered_data_telemetry
        endET = -1

        for i in range(len(telemetry_all)):
            telemetry = telemetry_all[i]
            scoring = scoring_all[i]

            if telemetry["mLastImpactET"] <= 0:
                telemetry["mLastImpactET"] = 0.0
            else:
                telemetry["mLastImpactET"] = scoring["mCurrentET"] - telemetry["mLastImpactET"]
            
            last_lap = scoring["mLastLapTime"]
            best_lap = scoring["mBestLapTime"]
            has_last_lap = 1.0 if last_lap > 0 else 0.0
            if last_lap <= 0.0:
                last_lap = 0  # placeholder
                best_lap = 0
            
            if scoring["mEndET"] < 0:
                endET = scoring_all[i+4]["mEndET"]

            else:
                endET = scoring["mEndET"]    
                

            data_state_per = [
                #dane do przewidzenia
                #dane ciągłe
                
                # scoring["mCurrLapTime"],
                
                round(scoring["mLapDist"]/scoring["mTotalLapDistance"],5),
                round(scoring["mCurrentET"]/endET,5),

                round(telemetry["mFuel"]/telemetry["mFuelCapacity"],5),
                round(telemetry['mWheel'][0]['mWear'], 5),  # Average wear across all four tires
                round(telemetry["mWheel"][1]["mWear"], 5),
                round(telemetry["mWheel"][2]["mWear"], 5),
                round(telemetry["mWheel"][3]["mWear"], 5),
                round(sum(telemetry["mWheel"][0]["mTemperature"])/len(telemetry["mWheel"][0]["mTemperature"]), 5),
                round(sum(telemetry["mWheel"][1]["mTemperature"])/len(telemetry["mWheel"][1]["mTemperature"]), 5),
                round(sum(telemetry["mWheel"][2]["mTemperature"])/len(telemetry["mWheel"][2]["mTemperature"]), 5),
                round(sum(telemetry["mWheel"][3]["mTemperature"])/len(telemetry["mWheel"][3]["mTemperature"]), 5),
                scoring["mAvgPathWetness"],

                #dane pomocnicze ciągłe
                telemetry["mLastImpactET"],
                telemetry["mLastImpactMagnitude"],
                scoring["mNumPenalties"],
                scoring["mRaining"],
                round(scoring["mAmbientTemp"], 2),
                round(scoring["mTrackTemp"], 2),

                #dane dyskretne pomocniczne
                
                
                telemetry["mDentSeverity"][0],  # Not defined which part of the car this refers to each index
                telemetry["mDentSeverity"][1],
                telemetry["mDentSeverity"][2], 
                telemetry["mDentSeverity"][3],
                telemetry["mDentSeverity"][4],
                telemetry["mDentSeverity"][5],
                telemetry["mDentSeverity"][6], 
                telemetry["mDentSeverity"][7],

                has_last_lap,
                scoring["mFinishStatus"],
                scoring["mTotalLaps"],
                scoring["mSector"],
                scoring["mNumPitstops"],
                int(scoring["mInPits"]),
                telemetry["mFrontTireCompoundIndex"],
                telemetry["multiplier"],

                #ciągłe nie używane do trenowania
                last_lap,
                best_lap,
                
                
        ]
            data_state.append(data_state_per)

       
        # with open("data/state_data.json", "w") as file:
        #     json.dump(data_state, file, indent=2)

        return data_state



def save_to_db(telem_file_raw, scoring_file_raw):
    # przykładowe stany i akcje
    states = extract_state(telem_file_raw, scoring_file_raw)

    # zamieniamy listy na JSON
    states_json = json.dumps(states)
   
    # tworzymy bazę / tabelę
    conn = sqlite3.connect("data/db_states_for_regress/race_data_states.db")
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
