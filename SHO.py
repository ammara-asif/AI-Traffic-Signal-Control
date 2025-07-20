import numpy as np
import matplotlib.pyplot as plt
import time
import traci
import Initialise as ini
import levy
import Fitness


"""Ensure sum of green times + yellow times = cycle_time, and green times stay within bounds."""
def enforce_cycle_time(solution, cycle_time, min_green, max_green, yellow_time=2):
    total_green = np.sum(solution)
    if total_green == 0:
        raise ValueError("Total green time cannot be zero.")

    scaling = (cycle_time - len(solution) * yellow_time) / total_green
    scaled = solution * scaling
    return np.clip(scaled, min_green, max_green)

def SHO_traffic_signal(pop, max_iter, min_green, max_green, phases, cycle_time):
    
    #Initialise population
    sea_horses = ini.initialize_traffic(pop, phases, min_green, max_green, cycle_time)
    sea_horses_fitness = np.zeros(pop)
   
    # Initial fitness evaluation
    for i in range(pop):
        sea_horses_fitness[i] = Fitness.f(sea_horses[i, :])
    sorted_indexes = np.argsort(sea_horses_fitness)

    #return values
    TargetPosition = sea_horses[sorted_indexes[0], :].copy()
    TargetFitness = sea_horses_fitness[sorted_indexes[0]]
    convergence_curve = np.zeros(max_iter)
    convergence_curve[0] = TargetFitness

    # Hyperparameters
    u = 0.1
    v = 0.1
    l = 0.1  

    t = 1
    while t < max_iter:
        print("ITERATION: ", t)

        # Movement behaviour 
        # Spiral motion-> local exploitation
        # Brownian motion-> exploration
        beta = np.random.randn(pop, phases)
        elite = np.tile(TargetPosition, (pop, 1))
        Step_length = levy.levy(pop, phases, 1.5) * 0.5  # Scaled Levy steps
        theta = np.random.rand(pop, phases) * 2 * np.pi
        row = u * np.exp(theta * v)
        x = row * np.cos(theta)
        y = row * np.sin(theta)
        z = row * theta

        spiral_component = Step_length * ((elite - sea_horses) * x * y * z + elite)  #Eq 4
        rand = np.random.rand(pop, phases)
        brownian_motion = rand * l * beta * (sea_horses - beta * elite)  #Eq 7
        r1 = np.random.rand(pop)
        Sea_horses_new1 = np.where(r1[:, None] > 0, sea_horses + spiral_component, sea_horses + brownian_motion)

        # Enforce constraints
        for i in range(pop):
            Sea_horses_new1[i, :] = np.clip(Sea_horses_new1[i, :], min_green, max_green)
            Sea_horses_new1[i, :] = enforce_cycle_time(Sea_horses_new1[i, :], cycle_time, min_green, max_green)

        # Predation behaviour 
        alpha = (1 - t / max_iter) ** (2 * t / max_iter) #Eq 11 
        r2 = np.random.rand(pop)
        rand = np.random.rand(pop, phases)
        success_part = alpha * (elite - rand * Sea_horses_new1) + (1 - alpha) * elite
        failure_part = (1 - alpha) * (Sea_horses_new1 - rand * elite) + alpha * Sea_horses_new1
        Sea_horses_new2 = np.where(r2[:, None] >=0.1, success_part, failure_part)  #Eq 10

        # Enforce constraints and evaluate fitness
        Sea_horsesFitness1 = np.zeros(pop)
        for i in range(pop):
            Sea_horses_new2[i, :] = enforce_cycle_time(Sea_horses_new2[i, :], cycle_time, min_green, max_green)
            Sea_horsesFitness1[i] = Fitness.f(Sea_horses_new2[i, :])

        # Breeding with mutation
        index = np.argsort(Sea_horsesFitness1)
        fathers = Sea_horses_new2[index[:pop//2], :] #Eq 12
        mothers = Sea_horses_new2[index[pop//2:], :]  
        r3 = np.random.rand(pop//2, 1)
        offspring = r3 * fathers + (1 - r3) * mothers #Eq 13

        # Mutation
        mutation_rate = 0.1
        mutation_strength = 0.1 * (max_green - min_green)
        offspring += mutation_rate * np.random.normal(0, mutation_strength, offspring.shape)
        for i in range(pop//2):
            offspring[i, :] = np.clip(offspring[i, :], min_green, max_green)
            offspring[i, :] = enforce_cycle_time(offspring[i, :], cycle_time, min_green, max_green)

        # Evaluate offspring fitness
        offspring_fitness = np.array([Fitness.f(ind) for ind in offspring])

        # Combine and select
        combined_pop = np.vstack((Sea_horses_new2, offspring))
        combined_fitness = np.concatenate((Sea_horsesFitness1, offspring_fitness))
        sorted_indices = np.argsort(combined_fitness)
        sea_horses = combined_pop[sorted_indices[:pop], :]
        current_best_fitness = combined_fitness[sorted_indices[0]]

        # Update target if improved
        if current_best_fitness < TargetFitness:
            TargetPosition = sea_horses[0, :].copy()
            TargetFitness = current_best_fitness

        convergence_curve[t] = TargetFitness
        t += 1

    # Round only the final solution
    TargetPosition = np.clip(TargetPosition, min_green, max_green)
    TargetPosition = enforce_cycle_time(TargetPosition, cycle_time, min_green, max_green)
    TargetPosition = np.round(TargetPosition).astype(int)

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
    tlsID = "J9" # traffic-light ID
   
    result = SHO_traffic_signal(popsize, max_iter, min_green, max_green, phases, cycle_time)

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
    #Replace the paths below with the full path to your own SUMO files (.sumocfg and sumo-gui.exe)
    config_file="data/test_sumo_visual.sumocfg"
    traci.start(["sumo-gui", "-c", config_file])
    logic = Fitness.build_custom_logic(ObjectivePosition)
    traci.trafficlight.setProgramLogic(tlsID, logic)

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        time.sleep(0.30)  
    traci.close()
