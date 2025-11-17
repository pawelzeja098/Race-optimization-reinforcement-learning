import numpy as np
# mLastImpactET: min=0.0, max=844.02001953125
# mLastImpactMagnitude: min=0.0, max=22007.326171875


# mDentSeverity[0]: min=0.0, max=2.0
# mDentSeverity[1]: min=0.0, max=2.0
# mDentSeverity[2]: min=0.0, max=0.0
# mDentSeverity[3]: min=0.0, max=2.0
# mDentSeverity[4]: min=0.0, max=2.0
# mDentSeverity[5]: min=0.0, max=2.0
# mDentSeverity[6]: min=0.0, max=0.0
# mDentSeverity[7]: min=0.0, max=2.0

def random_impact_magnitude(prob_impact = 0.001): #real is 0.01

    if np.random.rand() > prob_impact:
        return 0.0
    probabilities = np.load('E:/pracadyp/Race-optimization-reinforcement-learning/data/probabilities_impact/probabilities.npy')
    bin_edges = np.load('E:/pracadyp/Race-optimization-reinforcement-learning/data/probabilities_impact/bin_edges.npy')

    bin_idx = np.random.choice(len(probabilities), p=probabilities)
    low, high = bin_edges[bin_idx], bin_edges[bin_idx + 1]
    return np.random.uniform(low, high)

def generate_dent_severity(impact_magnitude,dent_severity_current):
    # Przykładowe progi i losowanie uszkodzenia
    if impact_magnitude == 0.0:
        return dent_severity_current
    values = [0, 1, 2, 3] #front,right,left,rear
    probabilities = [0.35, 0.15, 0.15, 0.35]

    random_value = np.random.choice(values, p=probabilities)

    #map elements to damage based on impact location
    if random_value == 0:
        elements_to_damage = [0,1,7]
    if random_value == 1:
        elements_to_damage = [1,3]
    if random_value == 2:
        elements_to_damage = [5,7]
    if random_value == 3:
        elements_to_damage = [3,4,5]
    
    if impact_magnitude < 2000:
        elem_damage_prob = 0.05
        damage_levels = [1.0]
        damage_probs = [1.0]
    elif impact_magnitude < 4000:
        elem_damage_prob = 0.08
        damage_levels = [1.0]
        damage_probs = [1.0]
    elif impact_magnitude < 6000:
        elem_damage_prob = 0.1
        damage_levels = [1.0, 2.0]
        damage_probs = [0.7, 0.3]
    elif impact_magnitude < 8000:
        elem_damage_prob = 0.2
        damage_levels = [1.0, 2.0]
        damage_probs = [0.6, 0.4]
    elif impact_magnitude < 12000:
        elem_damage_prob = 0.4
        damage_levels = [1.0, 2.0]
        damage_probs = [0.5, 0.5]
    elif impact_magnitude < 15000:
        elem_damage_prob = 0.6
        damage_levels = [1.0, 2.0]
        damage_probs = [0.4, 0.6]
    elif impact_magnitude < 20000:
        elem_damage_prob = 0.9
        damage_levels = [1.0, 2.0]
        damage_probs = [0.3, 0.7]
    else:
        elem_damage_prob = 1.0
        damage_levels = [2.0]
        damage_probs = [1.0]

    for i in elements_to_damage:
        if i == 2 or i == 6:
            continue  # te elementy się nie uszkadzają
        if np.random.rand() > elem_damage_prob:  # szansa na uszkodzenie tego elementu
            continue
        else:
            if dent_severity_current[i] < 2.0:
                dent_severity_current[i] += np.random.choice(damage_levels, p=damage_probs)
                dent_severity_current[i] = min(dent_severity_current[i], 2.0)  # max 2.0
    return dent_severity_current

random_impact_magnitude()