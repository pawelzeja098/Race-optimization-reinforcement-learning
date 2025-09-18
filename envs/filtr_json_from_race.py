# scoring_file_raw = "data/raw_races/scoring_data.json"
# telem_file_raw = "data/raw_races/telemetry_data.json"

scoring_file_filtered = "data/filtered_races/scoring_data.json"
telem_file_filtered = "data/filtered_races/telemetry_data.json"
import json

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

    for entry in raw_data_scoring:
        # for vehicle in raw_data_scoring.get("mVehicles", []):

        #     if vehicle.get("mIsPlayer"):
        vehicle = entry.get("mVehicles", [])

        wanted_weather_keys = ["mRaining","mAmbientTemp","mTrackTemp","mEndET", "mCurrentET"]
        subset_weather = {k: entry.get(k) for k in wanted_weather_keys}
        
        
        vehicle_sector = vehicle.get("mSector", -1)
        #save only if sector changed
        if vehicle_sector == curr_sector:
            print(f"curr_sector: {curr_sector}, vehicle_sector: {vehicle_sector}")
            continue
        print(f"Sector changed: {curr_sector} -> {vehicle_sector}")
        curr_sector = vehicle_sector
        wanted_keys = ["mLastLapTime","mBestLapTime","mCurrLapTime","mNumPitstops","mNumPenalties","mInPits"]
        subset_scoring_vehicle = {k: vehicle.get(k) for k in wanted_keys}

        # print(json.dumps(vehicle, indent=2))
        # scoring_records.append(data)
        filtered_data_scoring.append({**subset_scoring_vehicle, **subset_weather})

    for entry in raw_data_telemetry:

        wanted_keys = ["mFuel", "mFuelCapacity","mWheel","mDentSeverity","mFrontTireCompoundIndex","mCurrentSector","mLapNumber"]


        subset = {k: entry.get(k) for k in wanted_keys}

        subset["mWheel"] = [
                {
                    "mWear": wheel.get("mWear"),
                    "mBrakeTemp": wheel.get("mBrakeTemp"),
                    "mTemperature": wheel.get("mTemperature"),
                }
                for wheel in entry.get("mWheel", [])
            ]
        filtered_data_telemetry.append(subset)
        # print(json.dumps(subset, indent=2))
    

    return filtered_data_telemetry, filtered_data_scoring
    # with open(telem_file_filtered, "w") as f:
    #     json.dump(filtered_data_telemetry, f, indent=2)
    # with open(scoring_file_filtered, "w") as f:
    #     json.dump(filtered_data_scoring, f, indent=2)