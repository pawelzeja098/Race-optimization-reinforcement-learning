# mRaining: min=0.0, max=1.0
# mAmbientTemp: min=5.33, max=40.0
# mTrackTemp: min=9.0, max=47.35

from random import uniform, random, choice, randint

def generate_weather_conditions(num_conditions,mRaining_start=0.0,mAmbientTemp_start=None,mTrackTemp_start=None,no_rain=False):
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
    gap_timer = 0
    target_wetness = 0.0
    last_raining_value = mRaining
    next_temp = mAmbientTemp  # Na start cel jest taki sam jak obecna temp
    smoothing_speed = 0.005
    for _ in range(num_conditions):
        condition = {
            "mRaining": round(mRaining, 2),
            "mAmbientTemp": round(mAmbientTemp, 2),
            "mTrackTemp": round(mTrackTemp, 2),
            "mPathWetness": round(mPathWetness, 2)

        }
        weather_conditions.append(condition)

        if no_rain:
            mRaining = 0.0
            raining_change = 0.0
        else:

            if mRaining < 0.01:
                # Start deszczu
                if random() < 0.001: 
                    mRaining = uniform(0.1, 1.0)
            else:
                # Zmiana intensywności w trakcie deszczu
                if raining_time > 100 and random() < 0.01:
                    raining_change = choice([-0.1, -0.2, -0.05, 0.05, 0.1, 0.2])
                    mRaining = min(1.0, max(0.0, mRaining + raining_change))

                raining_time += 1
                
                # Koniec deszczu
                if random() < 0.005 and raining_time > 300: 
                    raining_time = 0
                    mRaining = 0.0

        #
        if mRaining != last_raining_value:
            target_wetness = mRaining
            last_raining_value = mRaining
            
            # Obliczamy opóźnienie TYLKO RAZ przy zmianie
            if target_wetness > mPathWetness:
               
                temp_factor = max(1, mAmbientTemp) 
                gap_timer = round(randint(2, 5) * temp_factor, 0)
                
            else:
               
                temp_factor = max(1, 50 - mAmbientTemp)
                gap_timer = round(randint(15, 25) * temp_factor, 0)  # Wydłużone z 5-10 do 15-25

        # 3. OBSŁUGA OPÓŹNIENIA (Step logic)
        if gap_timer > 0:
            gap_timer -= 1
        else:
            # Czas minął - następuje "skok" wartości (schodkowo)
            mPathWetness = target_wetness
        

        if abs(mAmbientTemp - next_temp) < 0.1:
    
    # Losujemy nowy cel
            if mAmbientTemp > 40:
                change = choice([-3, -2, -1, 0])      # Musi spadać
            elif mAmbientTemp < 9:
                change = choice([0, 1, 2, 3])          # Musi rosnąć
            else:
                change = choice([-2, -1, 0, 1, 2])     # Losowo
                
            next_temp = mAmbientTemp + change
        
     
        smoothing_speed = uniform(0.001, 0.008)

        # 2. Fizyka zmiany (LERP)
       
        mAmbientTemp += (next_temp - mAmbientTemp) * smoothing_speed

        # 3. Bezpiecznik (Clamp)
        mAmbientTemp = min(45, max(5, mAmbientTemp))

   

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
 
       

        # Hard limits
        mTrackTemp = min(47.35, max(9, mTrackTemp))


    return weather_conditions