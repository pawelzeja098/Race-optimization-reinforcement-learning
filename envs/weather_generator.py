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
                mRaining = 0.1
        else:
            # Jeśli pada, powolna zmiana intensywności
            mRaining = min(1, max(0, mRaining + uniform(-0.01, 0.01)))
            # Szansa na koniec deszczu tylko jeśli bardzo słaby
            if random() < 0.05: #mRaining < 0.05 
                mRaining = 0.0

        # Temperatura: powolne zmiany, ograniczone zakresy
        # mAmbientTemp += uniform(-0.2, 0.2)
        mAmbientTemp += choice([-0.2, 0.2])
        mAmbientTemp = min(40, max(5.33, mAmbientTemp))
        # Tor reaguje wolniej na zmiany otoczenia
        mTrackTemp += choice([-0.1, 0.1]) + 0.1 * (mAmbientTemp - mTrackTemp)
        mTrackTemp = min(47.35, max(9, mTrackTemp))

        # weather_conditions = {
        #     "mRaining": round(mRaining, 2),
        #     "mAmbientTemp": round(mAmbientTemp, 2),
        #     "mTrackTemp": round(mTrackTemp, 2)
        # }

    return weather_conditions