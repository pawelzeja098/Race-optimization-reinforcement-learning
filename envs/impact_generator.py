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

def random_impact_magnitude(prob_impact = 0.008,probabilities = None, bin_edges = None): 

    if np.random.rand() > prob_impact:
        return 0.0
    

    bin_idx = np.random.choice(len(probabilities), p=probabilities)
    low, high = bin_edges[bin_idx], bin_edges[bin_idx + 1]
    return np.random.uniform(low, high)


def generate_dent_severity(impact_magnitude):
    # Przykładowe progi i losowanie uszkodzenia
    if impact_magnitude == 0.0:
        return [0.0]*8
    values = [0, 1, 2, 3] #front,right,left,rear
    probabilities = [0.35, 0.15, 0.15, 0.35]
    dent_severity_change = [0.0]*8
    each_elem_damage_prob = [0.68,0.21,0.0,0.09,0.25,0.11,0.0,0.3]

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

    if impact_magnitude < 100:
        elem_damage_prob = 0.0
        damage_levels = [1.0]
        damage_probs = [1.0]
    elif impact_magnitude < 200:
        elem_damage_prob = 0.0005
        damage_levels = [1.0]
        damage_probs = [1.0]
    elif impact_magnitude < 500:
        elem_damage_prob = 0.001
        damage_levels = [1.0]
        damage_probs = [1.0]
    elif impact_magnitude < 1000:
        elem_damage_prob = 0.005
        damage_levels = [1.0]
        damage_probs = [1.0]
    elif impact_magnitude < 2000:
        elem_damage_prob = 0.1
        damage_levels = [1.0]
        damage_probs = [1.0]
    elif impact_magnitude < 4000:
        elem_damage_prob = 0.12
        damage_levels = [1.0]
        damage_probs = [1.0]
    elif impact_magnitude < 6000:
        elem_damage_prob = 0.18
        damage_levels = [1.0, 2.0]
        damage_probs = [0.7, 0.3]
    elif impact_magnitude < 8000:
        elem_damage_prob = 0.4
        damage_levels = [1.0, 2.0]
        damage_probs = [0.6, 0.4]
    elif impact_magnitude < 12000:
        elem_damage_prob = 0.7
        damage_levels = [1.0, 2.0]
        damage_probs = [0.5, 0.5]
    elif impact_magnitude < 15000:
        elem_damage_prob = 0.8
        damage_levels = [1.0, 2.0]
        damage_probs = [0.4, 0.6]
    elif impact_magnitude < 20000:
        elem_damage_prob = 0.95
        damage_levels = [1.0, 2.0]
        damage_probs = [0.3, 0.7]
    else:
        elem_damage_prob = 1.0
        damage_levels = [2.0]
        damage_probs = [1.0]

    for i in elements_to_damage:
        if i == 2 or i == 6:
            continue  # te elementy się nie uszkadzają
        
        final_elem_damage_prob = each_elem_damage_prob[i] * elem_damage_prob * 10

        if np.random.rand() > final_elem_damage_prob:  # szansa na uszkodzenie tego elementu
            continue
        else:
            dent_severity_change[i] = np.random.choice(damage_levels, p=damage_probs)
            
            # if dent_severity_current[i] < 2.0:
            #     dent_severity_current[i] += np.random.choice(damage_levels, p=damage_probs)
            #     dent_severity_current[i] = min(dent_severity_current[i], 2.0)  # max 2.0
    return dent_severity_change
# random_impact_magnitude()

import matplotlib.pyplot as plt
import numpy as np

# 1. Definicja funkcji modelu (z naprawioną logiką if/elif!)
def get_damage_model(impact_magnitude):
    # Domyślne wartości
    elem_damage_prob = 0.0
    prob_lvl_2 = 0.0 # Prawdopodobieństwo, że uszkodzenie będzie poz. 2 (jeśli wystąpi)

    if impact_magnitude < 100:
        elem_damage_prob = 0.005
        prob_lvl_2 = 0.0
    elif impact_magnitude < 200:
        elem_damage_prob = 0.01
        prob_lvl_2 = 0.0
    elif impact_magnitude < 500:
        elem_damage_prob = 0.02
        prob_lvl_2 = 0.0
    elif impact_magnitude < 1000:
        elem_damage_prob = 0.05
        prob_lvl_2 = 0.0
    elif impact_magnitude < 2000:
        elem_damage_prob = 0.1
        prob_lvl_2 = 0.0
    elif impact_magnitude < 4000:
        elem_damage_prob = 0.12
        prob_lvl_2 = 0.0
    elif impact_magnitude < 6000:
        elem_damage_prob = 0.18
        prob_lvl_2 = 0.3 # 30% szans na lvl 2
    elif impact_magnitude < 8000:
        elem_damage_prob = 0.4
        prob_lvl_2 = 0.4
    elif impact_magnitude < 12000:
        elem_damage_prob = 0.7
        prob_lvl_2 = 0.5
    elif impact_magnitude < 15000:
        elem_damage_prob = 0.8
        prob_lvl_2 = 0.6
    elif impact_magnitude < 20000:
        elem_damage_prob = 0.95
        prob_lvl_2 = 0.7
    else: # > 20000
        elem_damage_prob = 1.0
        prob_lvl_2 = 1.0

    # Obliczamy składowe prawdopodobieństwa dla wykresu
    # Całkowita szansa na uszkodzenie typu 2 = (szansa na uszkodzenie) * (udział typu 2)
    total_prob_lvl2 = elem_damage_prob * prob_lvl_2
    # Całkowita szansa na uszkodzenie typu 1 = reszta
    total_prob_lvl1 = elem_damage_prob - total_prob_lvl2
    
    return total_prob_lvl1, total_prob_lvl2

# # 2. Generowanie danych do wykresu
# x_values = np.linspace(0, 22000, 1000)
# y_lvl1 = []
# y_lvl2 = []

# for x in x_values:
#     p1, p2 = get_damage_model(x)
#     y_lvl1.append(p1)
#     y_lvl2.append(p2)

# y_lvl1 = np.array(y_lvl1)
# y_lvl2 = np.array(y_lvl2)

# # 3. Rysowanie wykresu
# plt.figure(figsize=(10, 6), dpi=300) # Wysoka rozdzielczość do druku
# plt.style.use('seaborn-v0_8-whitegrid') # Czysty styl akademicki

# # Wykres warstwowy (stacked)
# plt.stackplot(x_values, y_lvl2, y_lvl1, 
#               labels=['Uszkodzenie Krytyczne (Poziom 2)', 'Uszkodzenie Lekkie (Poziom 1)'],
#               colors=['#d62728', '#1f77b4'], # Czerwony dla krytycznych, Niebieski dla lekkich
#               alpha=0.6, step='post') # step='post' tworzy schodki

# # Dodatkowa grubasza linia sumaryczna na górze
# plt.step(x_values, y_lvl1 + y_lvl2, where='post', color='black', linewidth=1.5, linestyle='--', label='Całkowite Prawdopodobieństwo')

# # 4. Formatowanie
# plt.title('Model Prawdopodobieństwa Uszkodzeń', fontsize=14, pad=15)
# plt.xlabel('Siła Uderzenia [N]', fontsize=12)
# plt.ylabel('Prawdopodobieństwo Uszkodzenia', fontsize=12)

# plt.xlim(0, 22000)
# plt.ylim(0, 1.05)

# # Dodanie siatki i legendy
# plt.grid(True, which='both', linestyle='--', alpha=0.7)
# plt.legend(loc='upper left', frameon=True, facecolor='white', framealpha=1)

# # Oznaczenia osi Y jako procenty
# plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])

# # 5. Zapis i wyświetlenie
# plt.tight_layout()
# plt.savefig('model_uszkodzen.png') # Zapisze plik w folderze
# plt.show()