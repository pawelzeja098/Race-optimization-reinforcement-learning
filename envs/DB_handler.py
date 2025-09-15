import sqlite3
import json
import numpy as np
from generate_actions_BC import generate_actions_BC
from generate_state_BC import extract_state

def save_to_db(usage_multiplier = 3.0):
    # przykładowe stany i akcje
    states = extract_state()
    actions = generate_actions_BC()

    race_time = states[len(states) - 3][8]  # czas wyścigu

    # zamieniamy listy na JSON
    states_json = json.dumps(states)
    actions_json = json.dumps(actions)

    # tworzymy bazę / tabelę
    conn = sqlite3.connect("data/sqlite_db/race_data.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS races (
        race_id INTEGER PRIMARY KEY AUTOINCREMENT,
        usage_multiplier REAL,
        race_time REAL,
        states_json TEXT,
        actions_json TEXT
    )
    """)

    # zapisujemy rekord, nie podajemy race_id, SQLite automatycznie nada
    cursor.execute("""
    INSERT INTO races (usage_multiplier, race_time, states_json, actions_json)
    VALUES (?, ?, ?, ?)
    """, (usage_multiplier, race_time, states_json, actions_json))

    conn.commit()
    conn.close()

save_to_db(3.0)


def load_from_db():
    conn = sqlite3.connect("data/sqlite_db/race_data.db")
    cursor = conn.cursor()

    # wybierz wszystkie rekordy z tabeli
    cursor.execute("SELECT * FROM races")
    rows = cursor.fetchall()

    for row in rows:
        race_id, usage_multiplier, race_time, states_json, actions_json = row
        print(f"Race ID: {race_id}, usage_multiplier: {usage_multiplier}, race_time: {race_time}")
        
        # zamiana JSON z powrotem na listę
        states = json.loads(states_json)
        actions = json.loads(actions_json)
        
        print(f"Number of states: {len(states)}, Number of actions: {len(actions)}")
        print("Przykładowy state:", states)
        print("Przykładowa action:", actions)
        print("-"*40)

    conn.close()

load_from_db()

def delete_from_db(race_id):
    conn = sqlite3.connect("data/sqlite_db/race_data.db")
    cursor = conn.cursor()

    race_id_to_delete = 1
    cursor.execute("DELETE FROM races WHERE race_id = ?", (race_id_to_delete,))
    conn.commit()
    conn.close()

# delete_from_db(1)