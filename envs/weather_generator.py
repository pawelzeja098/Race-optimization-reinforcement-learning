# mRaining: min=0.0, max=1.0
# mAmbientTemp: min=5.33, max=40.0
# mTrackTemp: min=9.0, max=47.35

from random import uniform, random, choice

def generate_weather_conditions(num_conditions,mRaining_start=0.0,mAmbientTemp_start=None,mTrackTemp_start=None):
    weather_conditions = []
    mRaining = mRaining_start
    mAmbientTemp = mAmbientTemp_start if mAmbientTemp_start is not None else uniform(5, 40)
    mTrackTemp = mTrackTemp_start if mTrackTemp_start is not None else mAmbientTemp + uniform(2, 5)

    # if num_conditions == 1:
    raining_time = 0 
    for _ in range(num_conditions):
        condition = {
            "mRaining": round(mRaining, 2),
            "mAmbientTemp": round(mAmbientTemp, 2),
            "mTrackTemp": round(mTrackTemp, 2)
        }
        weather_conditions.append(condition)

        # Deszcz: jeśli nie pada, bardzo mała szansa na start
        if mRaining < 0.01:
            if random() < 0.001:  # 1% szansy na początek deszczu
                mRaining = uniform(0.1, 1.0)  
        else:
            # Jeśli pada, powolna zmiana intensywności
            mRaining = min(1, max(0, mRaining + uniform(-0.01, 0.01)))
            raining_time += 1
            if random() < 0.005 and raining_time > 300: #mRaining < 0.05 
                raining_time = 0
                mRaining = 0.0

        mAmbientTemp += uniform(-0.1, 0.1) 
        mAmbientTemp = min(40, max(5, mAmbientTemp))

        # --- 3. Temperatura Toru (FIZYKA) ---
        
        # Ustalanie "Celu" temperatury toru
        if mRaining > 0.05:
            # Gdy pada: woda chłodzi tor -> Tor dąży do temperatury otoczenia (lub nieco niżej przez parowanie)
            target_track_temp = mAmbientTemp - 2.0 
            inertia = 0.02 # Szybkie chłodzenie wodą
        else:
            # Gdy sucho: słońce grzeje -> Tor dąży do Ambient + SolarOffset
            # SolarOffset np. 10 stopni.
            target_track_temp = mAmbientTemp + 12.0 
            inertia = 0.005 # Asfalt nagrzewa/stygnie wolniej niż woda go chłodzi

        # Fizyka zbliżania się do celu (Smooth approach)
        # Nowa_temp = Stara_temp + (Cel - Stara_temp) * prędkość_zmiany + szum
        mTrackTemp += (target_track_temp - mTrackTemp) * inertia + uniform(-0.05, 0.05)

        # Hard limits
        mTrackTemp = min(47.35, max(9, mTrackTemp))

        # Temperatura: powolne zmiany, ograniczone zakresy
        # mAmbientTemp += uniform(-0.2, 0.2)
        # mAmbientTemp += choice([-0.2, 0.2])
        # mAmbientTemp = min(40, max(5.33, mAmbientTemp))
        # # Tor reaguje wolniej na zmiany otoczenia
        # if mRaining > 0.3:
        #     mTrackTemp += choice([-0.2, 0.05]) - 0.2 * (mTrackTemp - mAmbientTemp)
        # else:
        #     mTrackTemp += choice([-0.1, 0.1]) + 0.1 * (mAmbientTemp - mTrackTemp)
        # mTrackTemp = min(47.35, max(9, mTrackTemp))

        # weather_conditions = {
        #     "mRaining": round(mRaining, 2),
        #     "mAmbientTemp": round(mAmbientTemp, 2),
        #     "mTrackTemp": round(mTrackTemp, 2)
        # }

    return weather_conditions