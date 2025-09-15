
import numpy as np
import json

def extract_state(telemetry_json= "data/telemetry_data.json", scoring_json= "data/scoring_data.json"):
        data_state = []
        with open(scoring_json, "r") as file:
            scoring_all = json.load(file)
        with open(telemetry_json, "r") as file:
            telemetry_all = json.load(file)
        
        for i in range(len(telemetry_all)):
            telemetry = telemetry_all[i]
            scoring = scoring_all[i]
            if telemetry["mCurrentSector"] not in [0,1, 2]:
                if i != 0:
                    prev_sector = telemetry_all[i-1]["mCurrentSector"]
                    # zwiększamy o 1 i bierzemy modulo 3, żeby wracało do 0 po 2
                    telemetry["mCurrentSector"] = (prev_sector + 1) % 3
                else:
                    # dla pierwszego elementu, jeśli nieprawidłowy, ustaw na 0
                    telemetry["mCurrentSector"] = 0

            if telemetry["mCurrentSector"] == 0 or telemetry["mCurrentSector"] == 2:
                
                
            
                wear = 0
                for wheel in telemetry["mWheel"]:
                    # Process each wheel's telemetry data
                    #for now only wear
                    wear += wheel["mWear"]
                
                last_lap = scoring["mLastLapTime"]
                has_last_lap = 1.0 if last_lap > 0 else 0.0
                if last_lap <= 0.0:
                    last_lap = 10000.0  # placeholder
                
                    

                data_state_per = [
                    last_lap,
                    has_last_lap,
                    scoring["mBestLapTime"],
                    # scoring["mCurrLapTime"],
                    int(scoring["mInPits"]),
                    scoring["mNumPitstops"],
                    scoring["mRaining"],
                    round(scoring["mAmbientTemp"], 2),
                    round(scoring["mTrackTemp"], 2),
                    scoring["mEndET"],
                    scoring["mCurrentET"],

                    round(telemetry["mFuel"],2),
                    round(telemetry["mFuelCapacity"],2),
                    round(wear / 4.0, 2),  # Average wear across all four tires
                    telemetry["mDentSeverity"][0],  # Not defined which part of the car this refers to each index
                    telemetry["mDentSeverity"][1],
                    telemetry["mDentSeverity"][2], 
                    telemetry["mDentSeverity"][3],
                    telemetry["mDentSeverity"][4],
                    telemetry["mDentSeverity"][5],
                    telemetry["mDentSeverity"][6], 
                    telemetry["mDentSeverity"][7],

                    telemetry["mFrontTireCompoundIndex"],
                    3.0
                    
            ]
                data_state.append(data_state_per)

       
        # with open("data/state_data.json", "w") as file:
        #     json.dump(data_state, file, indent=2)

        return data_state

# extract_state()