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
                data = self.sock.recv(self.buffer_size).decode("utf-8")
                if not data:
                    break
                buffer += data

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

# if __name__ == "__main__":
#     client = TelemetryClient()
#     client.start()

#     import time
#     try:
#         while True:   # działa cały czas
#             data = client.get_latest()
#             if data:
#                 print(json.dumps(data, indent=2))  # drukuje CAŁY JSON ładnie sformatowany
#             time.sleep(1)  # 20 Hz, możesz zmienić
#     except KeyboardInterrupt:
#         print("\nZatrzymano klienta.")
#     finally:
#         client.stop()

if __name__ == "__main__":
    client = TelemetryClient()
    client.start()

    scoring_file = "scoring_data.json"
    telem_file = "telemetry_data.json"

    scoring_records = []
    telem_records = []

    try:
        scoring_saved = False
        while True:
            data = client.get_latest()
            
            if data:
                

                # zapis ScoringInfoV01
                if data.get("Type") == "ScoringInfoV01":
                    print(json.dumps(data, indent=2))
                    scoring_records.append(data)
                    with open(scoring_file, "w") as f:
                        json.dump(scoring_records, f, indent=2)

                    scoring_saved = True

                # zapis TelemInfoV01
                elif data.get("Type") == "TelemInfoV01" and scoring_saved:

                    telem_records.append(data)
                    print(json.dumps(data, indent=2))
                    with open(telem_file, "w") as f:
                        json.dump(telem_records, f, indent=2)
                    scoring_saved = False

            time.sleep(1)  # 20 Hz
    except KeyboardInterrupt:
        print("\nZatrzymano klienta.")
    finally:
        client.stop()
        print(f"Zapisano dane do {scoring_file} i {telem_file}")