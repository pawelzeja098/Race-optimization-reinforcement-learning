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
                    continue
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
            # self.queue.get(timeout=timeout)
            return self.queue.get(timeout=timeout)
        except Empty:
            return None

    def stop(self):
        self.running = False
        if self.sock:
            self.sock.close()


def collect_telemetry(usage_multiplier=1.0):
    print("Starting telemetry collection...")
    client = TelemetryClient()
    client.start()

    scoring_file = "data/raw_races/scoring_data.json"
    telem_file = "data/raw_races/telemetry_data.json"

    scoring_records = []
    telem_records = []

    try:
        scoring_saved = False
        i = 0
        scoring_counter = 0
        
        curr_sector = -1
        while True:
            data = client.get_latest()

            
            if data and data.get("Type") == "ScoringInfoV01":
                scoring_counter += 1
                if scoring_counter % 2 != 0:
                    continue  # zapisujemy co drugą paczkę ScoringInfoV01

                
                vehicles = data.get("mVehicles", [])

                # znajdź gracza
                player_vehicle = None
                for v in vehicles:
                    if v.get("mIsPlayer") == True:  # albo warunek po nazwie kierowcy
                        player_vehicle = v
                        break

                if player_vehicle:
                    # robimy kopię oryginalnego data
                    filtered_data = dict(data)
                    filtered_data["mVehicles"] = [player_vehicle]

                    while True:
                        try:
                            _ = client.queue.get_nowait()
                        except Empty:
                            break  # pusta kolejka → koniec czyszczenia

                    # szukamy najbliższego TelemInfoV01
                    telem = None
                    for _ in range(10):  # max 10 prób
                        msg = client.get_latest(timeout=0.1)
                        if msg and msg.get("Type") == "TelemInfoV01":
                            telem = msg
                            break
                        time.sleep(0.05)

                    if telem:
                        # scoring_records.append(filtered_data)
                        # telem_records.append(telem)
                        # i += 1

                        print (f"Pair {i} zapisane (Lap: {player_vehicle['mTotalLaps']}, Sector: {player_vehicle['mSector']})")

                        # dodajemy dopiero tutaj, w parze
                        telem["multiplier"] = usage_multiplier
                        scoring_records.append(filtered_data)
                        telem_records.append(telem)

                        i += 1

              
    except KeyboardInterrupt:
        print("\nZatrzymano klienta.")
    finally:
        with open(telem_file, "w") as f:
            json.dump(telem_records, f, indent=2)
        with open(scoring_file, "w") as f:
            json.dump(scoring_records, f, indent=2)
        client.stop()
        print(f"Zapisano dane do {scoring_file} i {telem_file}")

