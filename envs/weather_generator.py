# mRaining: min=0.0, max=1.0
# mAmbientTemp: min=5.33, max=40.0
# mTrackTemp: min=9.0, max=47.35

from random import uniform, random, choice, randint

def generate_weather_conditions(num_conditions,mRaining_start=0.0,mAmbientTemp_start=None,mTrackTemp_start=None):
    weather_conditions = []
    mRaining = mRaining_start
    mAmbientTemp = mAmbientTemp_start if mAmbientTemp_start is not None else uniform(5, 40)
    mTrackTemp = mTrackTemp_start if mTrackTemp_start is not None else mAmbientTemp + uniform(2, 5)
    mPathWetness = 0.0 if mRaining == 0.0 else uniform(0.1, 1.0)
    # if num_conditions == 1:
    raining_time = 0 
    gap_rain_wetness = 0
    raining_change_flag = False
    next_temp = mAmbientTemp
    how_quickly = 0
    how_quickly_track = 0
    target_track_temp = mTrackTemp
    for _ in range(num_conditions):
        condition = {
            "mRaining": round(mRaining, 2),
            "mAmbientTemp": round(mAmbientTemp, 2),
            "mTrackTemp": round(mTrackTemp, 2),
            "mPathWetness": round(mPathWetness, 2)

        }
        weather_conditions.append(condition)

        # Deszcz: jeśli nie pada, bardzo mała szansa na start
        if mRaining < 0.01:
            if random() < 0.001:  # 1% szansy na początek deszczu
                mRaining = uniform(0.1, 1.0)
                raining_change_flag = True  # <--- DODAJ TO
                raining_change = mRaining
                # Oblicz opóźnienie dla pierwszego deszczu (zanim tor zmoknie)
                # gap_rain_wetness = round(randint(5, 15) * (50 - mAmbientTemp), 0)
        else:
            # Jeśli pada, powolna zmiana intensywności
            if raining_time > 100 and random() < 0.01:
                raining_change = choice([-0.1,-0.2,-0.05, 0.05, 0.1, 0.2])
                mRaining = min(1, max(0, mRaining + raining_change))
                raining_change_flag = True
                

        
            raining_time += 1
         
            if random() < 0.005 and raining_time > 300: #mRaining < 0.05 
                raining_time = 0
                mRaining = 0.0
            
                mPathWetness = mRaining 
        if raining_change_flag:
            if raining_change > 0:
                gap_rain_wetness = round(randint(2, 8) * mAmbientTemp, 0)
            elif raining_change < 0:
                temp_factor = max(1, 50 - mAmbientTemp) # Zabezpieczenie, żeby nie zeszło poniżej 1
                gap_rain_wetness = round(randint(2, 8) * temp_factor, 0)
            
            if gap_rain_wetness > 0:
                gap_rain_wetness -= 1
            else:
                mPathWetness = mRaining
                raining_change_flag = False
        
        if mAmbientTemp == next_temp:
            next_temp = mAmbientTemp + choice([-2, -1, 0, 1, 2])
            how_quickly = randint(50, 200)
            change_per_step = round((next_temp - mAmbientTemp) / how_quickly, 2)

        mAmbientTemp += change_per_step
        mAmbientTemp = min(40, max(5, mAmbientTemp))

        # --- 3. Temperatura Toru (FIZYKA) ---
        
        # Ustalanie "Celu" temperatury toru
        # if mRaining > 0.05:
        #     # Gdy pada: woda chłodzi tor -> Tor dąży do temperatury otoczenia (lub nieco niżej przez parowanie)
        #     target_track_temp = mAmbientTemp - 2.0 
        #     inertia = 0.02 # Szybkie chłodzenie wodą
        # else:
        #     # Gdy sucho: słońce grzeje -> Tor dąży do Ambient + SolarOffset
        #     # SolarOffset np. 10 stopni.
        #     target_track_temp = mAmbientTemp + 12.0 
        #     inertia = 0.005 # Asfalt nagrzewa/stygnie wolniej niż woda go chłodzi

        if mTrackTemp == target_track_temp:
            if mTrackTemp < mAmbientTemp:
                target_track_temp += choice([0, 1, 2])
                if mPathWetness > 0.3:
                    target_track_temp += choice([-0.5, 0])
            else:
                target_track_temp += choice([-2, -1, 0])
                if mPathWetness > 0.3:
                    target_track_temp += choice([-0.5, 0])
            how_quickly_track = randint(400, 900)
        
        if how_quickly_track > 0:
            how_quickly_track -= 1
        else:
            mTrackTemp = target_track_temp
 
        # mTrackTemp += (target_track_temp - mTrackTemp) * inertia + uniform(-0.05, 0.05)

        # Hard limits
        mTrackTemp = min(47.35, max(9, mTrackTemp))


    return weather_conditions