import numpy as np

"""
Initialize population of traffic signal timing plans

Parameters:
    pop_size (int): Number of solutions in population
    n_phases (int): Number of traffic light phases (e.g., 4 for N/S/E/W)
    min_green (float or list): Minimum green time per phase (seconds)
    max_green (float or list): Maximum green time per phase (seconds)
    cycle_time (float): Total cycle time (including yellow/red clearance)

Returns:
    np.ndarray: Population array of shape (pop_size, n_phases)
"""

def initialize_traffic(pop_size, phases, min_green, max_green, cycle_time, yellow_time=2):
    
    # Calculate total available green time (subtract yellow times)
    total_yellow = phases * yellow_time
    total_green = cycle_time - total_yellow #90-8
    
    population = np.zeros((pop_size, phases), dtype=int)
    for i in range(pop_size):
        # Randomly distribute green times (Dirichlet ensures sum=1)
        green_times = np.random.dirichlet(np.ones(phases)) * total_green
        green_times = np.clip(green_times, min_green, max_green)
        
        green_int = np.round(green_times).astype(int) 
        diff = total_green - np.sum(green_int)
        while diff != 0:
            for j in range(phases):
                if diff == 0:
                    break
                if diff > 0 and green_int[j] < max_green:
                    green_int[j] += 1
                    diff -= 1
                elif diff < 0 and green_int[j] > min_green:
                    green_int[j] -= 1
                    diff += 1

        population[i] = green_int
        print(population[i])
    return population


