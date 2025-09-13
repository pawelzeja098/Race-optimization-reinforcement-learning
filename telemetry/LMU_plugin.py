import socket
import threading
import json
from queue import Queue, Empty
import time

class TelemetryClient:
    def __init__(self, host="127.0.0.1", port=5000, buffer_size=4096):
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.sock = None
        self.running = False
        self.queue = Queue()

    def start(self):
        """Start telemetry client in background thread"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        self.running = True

        thread = threading.Thread(target=self._receive_loop, daemon=True)
        thread.start()

    def _receive_loop(self):
        buffer = ""
        while self.running:
            try:
                # data = self.sock.recv(self.buffer_size).decode("utf-8")
                # if not data:
                #     break
                # buffer += data
                data = self.sock.recv(self.buffer_size)
                if not data:
                    break
                try:
                    text = data.decode("utf-8", errors="ignore")
                except Exception as e:
                    print("Decode error:", e)
                    continue
                buffer += text

                # Plugin prawdopodobnie wysyła JSONy po \n
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    try:
                        msg = json.loads(line)
                        self.queue.put(msg)
                    except json.JSONDecodeError:
                        pass
            except Exception as e:
                print("Telemetry error:", e)
                self.running = False

    def get_latest(self, timeout=0.1):
        """Zwraca ostatnią paczkę telemetryczną (lub None)"""
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None

    def stop(self):
        self.running = False
        if self.sock:
            self.sock.close()


if __name__ == "__main__":
    client = TelemetryClient()
    client.start()

    scoring_file = "data/scoring_data.json"
    telem_file = "data/telemetry_data.json"

    scoring_records = []
    telem_records = []

    try:
        scoring_saved = False
        i = 0
        curr_sector = -1
        while True:  # Limit to 100 iterations
            data = client.get_latest()
            
            if data:
                

                # zapis ScoringInfoV01
                if data.get("Type") == "ScoringInfoV01":
                    print(f"ScoringInfoV01 {i}")
                    for vehicle in data.get("mVehicles", []):

                        if vehicle.get("mIsPlayer"):
                            print("Found player vehicle")
                            wanted_weather_keys = ["mRaining","mAmbientTemp","mTrackTemp","mEndET", "mCurrentET"]
                            subset_weather = {k: data.get(k) for k in wanted_weather_keys}
                            
                            
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
                            scoring_records.append({**subset_scoring_vehicle, **subset_weather})
                            
                            
                            scoring_saved = True

                # zapis TelemInfoV01
                elif data.get("Type") == "TelemInfoV01" and scoring_saved:

                    

                    wanted_keys = ["mFuel", "mFuelCapacity","mWheel","mDentSeverity","mFrontTireCompoundIndex","mCurrentSector"]


                    subset = {k: data.get(k) for k in wanted_keys}

                    subset["mWheel"] = [
                            {
                                "mWear": wheel.get("mWear"),
                                "mBrakeTemp": wheel.get("mBrakeTemp"),
                                "mTemperature": wheel.get("mTemperature"),
                            }
                            for wheel in data.get("mWheel", [])
                        ]
                    telem_records.append(subset)
                    print(json.dumps(subset, indent=2))
                    
                    scoring_saved = False
                    i += 1
            # time.sleep(0.05)  # 20 Hz
    except KeyboardInterrupt:
        print("\nZatrzymano klienta.")
    finally:
        with open(telem_file, "w") as f:
            json.dump(telem_records, f, indent=2)
        with open(scoring_file, "w") as f:
            json.dump(scoring_records, f, indent=2)
        client.stop()
        print(f"Zapisano dane do {scoring_file} i {telem_file}")