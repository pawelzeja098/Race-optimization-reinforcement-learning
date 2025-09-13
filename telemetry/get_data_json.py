import json

with open("scoring_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Jeśli plik to lista obiektów (np. kolejne okrążenia)
for race in data:
    for vehicle in race.get("mVehicles", []):
        if vehicle.get("mIsPlayer"):
            # Zwraca pierwszy znaleziony pojazd gracza
            print(json.dumps(vehicle, indent=2, ensure_ascii=False))
            break
    else:
        continue
    break