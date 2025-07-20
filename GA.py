import numpy as np
import matplotlib.pyplot as plt
import time
import traci
import Initialise as ini
import Fitness
import random


"""Ensure sum of green times + yellow times = cycle_time, and green times stay within bounds."""
def enforce_cycle_time(solution, cycle_time, min_green, max_green, yellow_time=2):
    total_green = np.sum(solution)
    if total_green == 0:
        raise ValueError("Total green time cannot be zero.")

    scaling = (cycle_time - len(solution) * yellow_time) / total_green
    scaled = solution * scaling
    return np.clip(scaled, min_green, max_green)

def mutate(individual, mutation_rate, min_green, max_green):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            mutation_amount = random.uniform(-1, 1)
            individual[i] += mutation_amount
            individual[i] = max(min(individual[i], max_green), min_green)
    individual = enforce_cycle_time(np.array(individual), cycle_time, min_green, max_green)
    return individual

def GA_traffic_signal(pop,max_iter,min_green,max_green,phases, cycle_time):

    #initialize population
    population = ini.initialize_traffic(pop, phases, min_green, max_green, cycle_time)
    convergence_curve = np.zeros(max_iter)
    fitness = np.zeros(pop)
    fitness = Fitness.evaluate_fitness(population)

    # Find initial best solution
    TargetPosition = population[np.argmin(fitness)].copy() 
    TargetFitness = np.min(fitness)
    convergence_curve[0] = TargetFitness
    
    #algo paramters
    tournament_size=3
    mutation_rate=0.2
    
    for t in range(1, max_iter):
        print("ITERATION: ", t)
        
        # I. Selection for reproduction (tournament)
        selected = []
        for _ in range(pop):
            tournament_indices = np.random.choice(pop, size=tournament_size, replace=False)
            winner_index = tournament_indices[np.argmin(fitness[tournament_indices])]
            selected.append(population[winner_index])
        selected = np.array(selected)

        offspring=[]
        for i in range(pop//2):
             # II. Crossover
            parent1, parent2 = selected[i], selected[i + 1]
            alpha = np.random.random()
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = alpha * parent2 + (1 - alpha) * parent1
            
            # Enforce total cycle time constraint
            child1 = enforce_cycle_time(child1, cycle_time, min_green, max_green)
            child2 = enforce_cycle_time(child2, cycle_time, min_green, max_green)
           
            # III. Mutation
            child1 = mutate(child1, mutation_rate, min_green, max_green)
            child2 = mutate(child2, mutation_rate, min_green, max_green)
            offspring.extend([child1, child2])

        offspring_fitness = np.array([Fitness.f(ind) for ind in offspring])
        
        # Combine populations
        combined_population = np.vstack((population, np.array(offspring)))
        combined_fitness = np.concatenate((fitness, offspring_fitness))
        
        # Selection 
        sorted_indexes = np.argsort(combined_fitness)
        population = combined_population[sorted_indexes[:pop]]
        fitness = combined_fitness[sorted_indexes[:pop]]
        
       # Update best solution
        current_min_fitness = np.min(fitness)
        if current_min_fitness < TargetFitness:
            TargetFitness = current_min_fitness
            TargetPosition = population[np.argmin(fitness)]

        
        convergence_curve[t] = TargetFitness        
        TargetPosition = [round(num) for num in TargetPosition]  # Round to integers
        TargetPosition = [max(min(num, max_green), min_green) for num in TargetPosition]
        
    
    
    return {
        'TargetFitness': TargetFitness,
        'TargetPosition': TargetPosition,
        'Convergence_curve': convergence_curve,
    }


if __name__=="__main__":
    
    start_time = time.time()
    popsize=30
    max_iter=40
    min_green= 10
    max_green=60
    phases=4
    cycle_time=90
    tlsID = "J9"    # traffic-light ID
   
    result = GA_traffic_signal(popsize, max_iter, min_green, max_green, phases, cycle_time)

    ObjectiveFitness = result['TargetFitness']
    ObjectivePosition = result['TargetPosition']
    Convergence_curve = result['Convergence_curve']
    
    elapsed_time = time.time() - start_time

    #Plotting
    plt.semilogy(range(1, max_iter+1), Convergence_curve, color='r', linewidth=2.5)
    plt.title('Convergence curve')
    plt.xlabel('Iteration')
    plt.ylabel('Best score obtained so far')
    plt.show()

    
    #Displaying
    print(f'The algo running time is: {elapsed_time:.4f} seconds')
    print(f'The best wait times: {ObjectiveFitness} seconds')
    print(f'Avg wait time per car: {ObjectiveFitness/50} seconds')
    print(f'The best green times for phases: {ObjectivePosition}')

    #Simulating
    config_file="D:\\user\\projects\\AI_project\\test_sumo_visual.sumocfg"
    traci.start(["C:\\Program Files (x86)\\Eclipse\\Sumo\\bin\\sumo-gui.exe", "-c", config_file])
    logic = Fitness.build_custom_logic(ObjectivePosition)
    traci.trafficlight.setProgramLogic(tlsID, logic)

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        time.sleep(0.3)  
    traci.close()
