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

def random_impact_magnitude(prob_impact = 0.01):

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
    
    if impact_magnitude < 2000:
        elem_damage_prob = 0.01
        damage_levels = [1.0]
        damage_probs = [1.0]
    elif impact_magnitude < 4000:
        elem_damage_prob = 0.015
        damage_levels = [1.0]
        damage_probs = [1.0]
    elif impact_magnitude < 6000:
        elem_damage_prob = 0.02
        damage_levels = [1.0, 2.0]
        damage_probs = [0.7, 0.3]
    elif impact_magnitude < 8000:
        elem_damage_prob = 0.03
        damage_levels = [1.0, 2.0]
        damage_probs = [0.6, 0.4]
    elif impact_magnitude < 12000:
        elem_damage_prob = 0.04
        damage_levels = [1.0, 2.0]
        damage_probs = [0.5, 0.5]
    elif impact_magnitude < 15000:
        elem_damage_prob = 0.05
        damage_levels = [1.0, 2.0]
        damage_probs = [0.4, 0.6]
    elif impact_magnitude < 20000:
        elem_damage_prob = 0.07
        damage_levels = [1.0, 2.0]
        damage_probs = [0.3, 0.7]
    else:
        elem_damage_prob = 0.09
        damage_levels = [2.0]
        damage_probs = [1.0]

    for i in range(8):
        if i == 2 or i == 6:
            continue  # te elementy się nie uszkadzają
        if np.random.rand() < elem_damage_prob:  # szansa na uszkodzenie tego elementu
            continue
        else:
            if dent_severity_current[i] < 2.0:
                dent_severity_current[i] += np.random.choice(damage_levels, p=damage_probs)
                dent_severity_current[i] = min(dent_severity_current[i], 2.0)  # max 2.0
    return dent_severity_current

random_impact_magnitude()