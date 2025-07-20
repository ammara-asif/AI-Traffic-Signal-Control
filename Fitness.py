import traci
import traci.constants as tc
import numpy as np

#Calculate fitness (wait times) of each individual by simulating in SUMO

def build_custom_logic(green_times, yellow=2):
    masks = ["Grrr", "rGrr", "rrGr", "rrrG"]
    phases1 = []
    for idx, g in enumerate(green_times):
        # 1) green phase
        phases1.append(traci.trafficlight.Phase(duration=int(g), state=masks[idx]))
        # 2) yellow clearance (after green)
        phases1.append(traci.trafficlight.Phase(duration=yellow, state=masks[idx].replace("G", "y")))
    return traci.trafficlight.Logic("my", 0, 0, phases1)

def run_simulation(individual,
                    sumo=("sumo"),
                    config_file="data/test_sumo_visual.sumocfg",
                    max_steps=1000):
    
    traci.start([sumo, "-c", config_file, "--start", "--no-step-log", "true", "--duration-log.disable", "true"])
    #tlsID = traci.trafficlight.getIDList()[0]
    tlsID= "J9"
    custom_logic = build_custom_logic( individual)
    traci.trafficlight.setProgramLogic(tlsID, custom_logic)
    
    total_wait = 0.0
    vehicle_tracker = {}  # {vehicle_id: last_waiting_time}
    
    for _ in range(max_steps):
        traci.simulationStep()
        
        current_vehicles = traci.vehicle.getIDList()
        
        # Track new vehicles
        for vid in current_vehicles:
            if vid not in vehicle_tracker:
                vehicle_tracker[vid] = 0.0
        
        # Calculate incremental waiting time
        for vid in list(vehicle_tracker.keys()): 
            if vid in current_vehicles:
                current_wait = traci.vehicle.getWaitingTime(vid)
                total_wait += max(0, current_wait - vehicle_tracker[vid])
                vehicle_tracker[vid] = current_wait
            else:
                # Vehicle left - add final waiting time
                total_wait += vehicle_tracker.pop(vid)

    # Add remaining vehicles' waiting times
    for vid in vehicle_tracker:
        total_wait += vehicle_tracker[vid]

    traci.close()
    #print("Simulated", total_wait)
    return total_wait

def f(individual):
    try:
        return run_simulation(individual)

    except Exception as e:
        print(f"[ERROR] Simulation failed: {e}")    
        if traci.isLoaded():
            traci.close()
        return float("inf")
    
def evaluate_fitness(population):
    return np.array([f(ind) for ind in population])
    
