# mRaining: min=0.0, max=1.0
# mAmbientTemp: min=5.33, max=40.0
# mTrackTemp: min=9.0, max=47.35

from random import uniform, random

def generate_weather_conditions(num_conditions):
    weather_conditions = []
    mRaining = 0.0  # start: nie pada
    for _ in range(num_conditions):
        condition = {
            "mRaining": round(mRaining, 2),
            "mAmbientTemp": round(uniform(5, 40), 2),
            "mTrackTemp": round(uniform(9, 47), 2)
        }
        weather_conditions.append(condition)
        # Jeśli nie pada, mała szansa na deszcz
        if mRaining < 0.01:
            if random() < 0.02:  # 2% szansy na deszcz
                mRaining = 0.1 # zaczyna padać
            else:
                mRaining = 0.0
        else:
            # Jeśli pada, płynna zmiana ±0.02
            mRaining = min(1, max(0, mRaining + uniform(-0.02, 0.02)))
    return weather_conditions