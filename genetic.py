import os
import sys
import random
import numpy as np
import libsumo as traci
import csv
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from multiprocessing import Pool
from datetime import datetime
from lxml import etree

# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true", help="Run the genetic learning algorithm")
parser.add_argument("--plot", action="store_true", help="Plotting the metrics compared to baseline")
parser.add_argument("--baseline", action="store_true", help="Run baseline fixed-speed agent")
parser.add_argument('--scenario', type=str, default='gordo', help='Scenario name')
parser.add_argument('--weather', type=str, default='osm_clear_dry', help='Weather condition')
args = parser.parse_args()

# Gotta hook SUMO tools in here if SUMO_HOME is set, otherwise stop right there
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# SIMULATION
SCENARIO = args.scenario
WEATHER = args.weather

# Basic SUMO config we're feeding to traci
SUMO_CONFIG = [
    "sumo",  # use 'sumo' for no GUI
    "-c", f"{SCENARIO}/{WEATHER}.sumocfg",
    "--step-length", "1.0",
    "--scale", "1.0",
    "--quit-on-end", "True",
    "--start",
    "--delay", "0",
    "--no-step-log", "True",
    "--no-warnings", "True"
]
SPEED_MIN = 1
SPEED_MAX = 100

# Hyperparameters for genetic algorithm
POPULATION_SIZE = 30
TOTAL_GENERATIONS = 200
MAX_STEPS_PER_EPISODE = 1000
D = 0.25

LOG_FOLDER = f"results/{SCENARIO}_{WEATHER}"

# Breeding new population odds
P_RANDOM_MUTATION = .99
RANDOM_MUTAION_DECAY = .9

# FITNESS WEIGHTS
JAM = -.1
EBRAKES = -.3
COLLISIONS = -.55
AVG_TRIP_TIME = 0
WAIT_TIME = 0
TRIP_COUNT = .1

# Spawn vehicles all over the place with random routes
def spawn_vehicles(num_vehicles=100):
    edges = [e for e in traci.edge.getIDList() if not e.startswith(":")]
    for i in range(num_vehicles):
        from_edge = random.choice(edges)
        to_edge = random.choice(edges)
        if from_edge != to_edge:
            route_id = f"route_{i}"
            veh_id = f"veh_{i}"
            try:
                traci.route.add(route_id, [from_edge, to_edge])
                traci.vehicle.add(veh_id, route_id)
            except Exception as e:
                print(f"Failed to add vehicle {veh_id}: {e}")

def new_individual(parent1, parent2):
    # crossover
    b = np.random.uniform(-D, 1+D, (parent1.shape))
    individual = parent1*b + parent2*(1-b)
    # random mutation
    rand_array = np.random.rand(parent1.shape[0])
    x = np.array(range(parent1.shape[0]), dtype=float)
    individual = np.piecewise(x, [rand_array[x.astype(int)] < P_RANDOM_MUTATION],
                              [lambda x: random.randrange(SPEED_MIN, SPEED_MAX), lambda x: individual[x.astype(int)]])
    return individual

# https://www.scirp.org/journal/paperinformation?paperid=51987
def new_population(old_population, fitness):
    # roulette wheel selection: pick parents for each individual based w/ fitness weighted prob
    population = np.zeros(old_population.shape)
    if np.min(fitness) < 0:
        fitness += -np.min(fitness)
    print(fitness)
    total_fitness = np.sum(fitness)
    prob_array = np.zeros(POPULATION_SIZE)
    prob_sum = 0
    for i in range(POPULATION_SIZE):
        print((fitness[i]/total_fitness))
        prob_array[i] = prob_sum + (fitness[i]/total_fitness)
        prob_sum = prob_array[i]
    # elitism: best from last generation gets free pass into next
    population[0] = old_population[np.argmax(fitness)]
    for i in range(1,POPULATION_SIZE):
        p1 = np.random.rand()
        idx_1 = 0
        p2 = np.random.rand()
        idx_2 = 0
        for j in range(len(prob_array)-1, -1, -1):
            if prob_array[j] > p1:
                idx_1 = j
            if prob_array[j] > p2:
                idx_2 = j
        population[i] = new_individual(old_population[idx_1], old_population[idx_2])

    return population

def start_population(): # set speed limits between 0 and 100
    traci.start(SUMO_CONFIG)
    numEdges = 0

    for edge in traci.edge.getIDList():
        if not edge.startswith(":"):
            print(edge, end=",")
            numEdges+=1
    traci.close()
    return np.random.rand(POPULATION_SIZE, numEdges)*100

def set_speed_limits(individual):
    i = 0
    for edge in traci.edge.getIDList():
        if not edge.startswith(":"):
            traci.edge.setMaxSpeed(edge, individual[i])
            i+=1

def fitness(jams, ebrakes, collisions, totalTravelTime, waiting_time, tripCount, baseline):
    if tripCount==0: tripCount=1

    jams = (jams - baseline["jams"]) / baseline["jams"]
    ebrakes = (ebrakes - baseline["ebrakes"]) / baseline["ebrakes"]
    collisions = (collisions - baseline["collisions"]) / baseline["collisions"]
    totalTravelTime = (totalTravelTime - baseline["totalTravelTime"]) / baseline["totalTravelTime"]
    waiting_time = (waiting_time - baseline["waitingTime"]) / baseline["waitingTime"]
    tripCount = (tripCount - baseline["tripCount"]) / baseline["tripCount"]
    if tripCount == 0: tripCount = 1

    return JAM*jams + EBRAKES*ebrakes +COLLISIONS*collisions + AVG_TRIP_TIME*(totalTravelTime/tripCount) + WAIT_TIME*waiting_time + TRIP_COUNT*tripCount

# Record  stats
def log_metrics(filename, gen, num, fit, jams, ebrakes, collisions, totalTravelTime, waitingTime, tripcount):
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([gen, num, fit, jams, ebrakes, collisions, totalTravelTime, waitingTime, tripcount])

def log_genes(filename, gen, population):
    with open(filename, 'a', newline='') as f:
        for i in range(len(population)):
            writer = csv.writer(f)
            writer.writerow([gen, i, *population[i]])

def run_individual(individual, gen, num, baseline):
    id = f"g{gen}_{num}"
    traci.start(SUMO_CONFIG+["--statistic-output", f"{LOG_FOLDER}/{id}.xml"])
    set_speed_limits(individual)
    # spawn_vehicles(num_vehicles=500)
    traci.simulationStep(MAX_STEPS_PER_EPISODE)
    traci.close()

    # get log file
    tree = etree.parse(f"{LOG_FOLDER}/{id}.xml")
    root = tree.getroot()

    # divide by steps to normalize
    jams = int(root.find('teleports').attrib['total'])
    ebraking = int(root.find('safety').attrib['emergencyBraking'])
    collisions = int(root.find('safety').attrib['collisions'])
    totalTravelTime = float(root.find('vehicleTripStatistics').attrib['totalTravelTime'])
    waitingTime = float(root.find('vehicleTripStatistics').attrib['waitingTime'])
    tripCount = int(root.find('vehicleTripStatistics').attrib['count'])

    os.remove(f"{LOG_FOLDER}/{id}.xml") # for space concerns
    fit = fitness(jams, ebraking, collisions, totalTravelTime, waitingTime, tripCount, baseline)
    return np.array((gen, num, fit, jams, ebraking, collisions, totalTravelTime, waitingTime, tripCount))

def run_baseline(speed = 25):
    try:
        os.mkdir(LOG_FOLDER)
    except OSError as error:
        print("log folder already exists. data will be overwritten")

    traci.start(SUMO_CONFIG)

    numEdges = 0
    for edge in traci.edge.getIDList():
        if not edge.startswith(":"):
            print(edge, end=",")
            numEdges+=1
    traci.close()

    speed_arr = []
    for i in range(numEdges):
        speed_arr.append(speed)

    traci.start(SUMO_CONFIG+["--statistic-output", f"{LOG_FOLDER}/baseline.xml"])
    set_speed_limits(speed_arr)
    traci.simulationStep(MAX_STEPS_PER_EPISODE)
    traci.close()

    # get log file
    tree = etree.parse(f"{LOG_FOLDER}/baseline.xml")
    root = tree.getroot()

    jams = int(root.find('teleports').attrib['total']) + 1
    ebraking = int(root.find('safety').attrib['emergencyBraking']) + 1
    collisions = int(root.find('safety').attrib['collisions']) + 1
    totalTravelTime = float(root.find('vehicleTripStatistics').attrib['totalTravelTime']) + 1
    waitingTime = float(root.find('vehicleTripStatistics').attrib['waitingTime']) + 1
    tripCount = int(root.find('vehicleTripStatistics').attrib['count']) + 1


    fit = 0

    log_metrics(f"{LOG_FOLDER}/baseline_data.csv", "baseline", "baseline", fit, jams, ebraking, collisions, totalTravelTime, waitingTime, tripCount)

def train():
    try:
        os.mkdir(LOG_FOLDER)
    except OSError as error:
        print("log folder already exists. data will be overwritten")
    base_df = pd.read_csv(f"results/{SCENARIO}_{WEATHER}/baseline_data.csv", header=None,
                          names=["gen", "num", "fit", "jams", "ebrakes", "collisions",
                                 "totalTravelTime", "waitingTime", "tripCount"])
    baseline = base_df.iloc[0]

    global P_RANDOM_MUTATION
    population = start_population()

    for g in range(TOTAL_GENERATIONS):
        print(f"############## GENERATION {g} ##############")

        pool = Pool(processes=POPULATION_SIZE)
        return_data = np.array([pool.apply(run_individual, args=(population[i], g, i, baseline,)) for i in range(POPULATION_SIZE)])

        for d in return_data:
            log_metrics(f"{LOG_FOLDER}/metric_data.csv", *d)
        log_genes(f"{LOG_FOLDER}/gene_data.csv", g, population)


        population = new_population(population, return_data[:,2])
        P_RANDOM_MUTATION = RANDOM_MUTAION_DECAY * P_RANDOM_MUTATION

def plot_metrics():
    train_df = pd.read_csv(f"compiled_results/{SCENARIO}_{WEATHER}_metric.csv", header=None,
                           names=["gen", "num", "fit", "jams", "ebrakes", "collisions",
                                  "totalTravelTime", "waitingTime", "tripCount"])

    try:
        base_df = pd.read_csv(f"compiled_results/{SCENARIO}_{WEATHER}_baseline.csv", header=None,
                              names=["gen", "num", "fit", "jams", "ebrakes", "collisions",
                                     "totalTravelTime", "waitingTime", "tripCount"])
        baseline = base_df.iloc[0]
        avgTravelTime = baseline["totalTravelTime"] / baseline["tripCount"] if baseline["tripCount"] > 0 else 0
    except FileNotFoundError:
        print("Baseline data not found. Skipping baseline overlay.")
        baseline = None
        avgTravelTime = None

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 4, 1)
    plt.scatter(train_df["gen"], train_df["fit"], label="Trained Agent", s=5, alpha=0.33)
    if baseline is not None:
        plt.axhline(y=baseline["fit"], color='r', linestyle='--', label="Baseline")
    plt.title("Fitness per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()

    plt.subplot(2, 4, 2)
    plt.scatter(train_df["gen"], train_df["collisions"], color='g', s=5, alpha=0.33)
    if baseline is not None:
        plt.axhline(y=baseline["collisions"], color='r', linestyle='--', label="Baseline")
    plt.title("Collisions per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Collisions")
    plt.legend()

    plt.subplot(2, 4, 3)
    plt.scatter(train_df["gen"], train_df["waitingTime"], color='orange', s=5, alpha=0.33)
    if baseline is not None:
        plt.axhline(y=baseline["waitingTime"], color='r', linestyle='--', label="Baseline")
    plt.title("Waiting Time per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Waiting Time")
    plt.legend()

    plt.subplot(2, 4, 4)
    plt.scatter(train_df["gen"], train_df["ebrakes"], color='yellow', s=5, alpha=0.33)
    if baseline is not None:
        plt.axhline(y=baseline["ebrakes"], color='r', linestyle='--', label="Baseline")
    plt.title("Emergency Braking per Generation")
    plt.xlabel("Generation")
    plt.ylabel("E-Brakes")
    plt.legend()

    plt.subplot(2, 4, 5)
    plt.scatter(train_df["gen"], train_df["jams"], color='purple', s=5, alpha=0.33)
    if baseline is not None:
        plt.axhline(y=baseline["jams"], color='r', linestyle='--', label="Baseline")
    plt.title("Teleports (Jams) per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Jams")
    plt.legend()

    plt.subplot(2, 4, 6)
    plt.scatter(train_df["gen"], train_df["totalTravelTime"] / train_df["tripCount"], color='blue', s=5, alpha=0.33)
    if avgTravelTime is not None:
        plt.axhline(y=avgTravelTime, color='r', linestyle='--', label="Baseline")
    plt.title("Avg Travel Time per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Avg Travel Time")
    plt.legend()

    plt.subplot(2, 4, 7)
    plt.scatter(train_df["gen"], train_df["tripCount"], color='cyan', s=5, alpha=0.33)
    if baseline is not None:
        plt.axhline(y=baseline["tripCount"], color='r', linestyle='--', label="Baseline")
    plt.title("Trip Count per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Trip Count")
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_weights():
    train_df = pd.read_csv(f"compiled_results/{SCENARIO}_{WEATHER}_metric.csv", header=None,
                           names=["gen", "num", "fit", "jams", "ebrakes", "collisions",
                                  "totalTravelTime", "waitingTime", "tripCount"])

    try:
        base_df = pd.read_csv(f"compiled_results/{SCENARIO}_{WEATHER}_baseline.csv", header=None,
                              names=["gen", "num", "fit", "jams", "ebrakes", "collisions",
                                     "totalTravelTime", "waitingTime", "tripCount"])
        baseline = base_df.iloc[0]
        avgTravelTime = baseline["totalTravelTime"] / baseline["tripCount"] if baseline["tripCount"] > 0 else 0
    except FileNotFoundError:
        print("Baseline data not found. Skipping baseline overlay.")
        baseline = None
        avgTravelTime = None

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 4, 1)
    plt.scatter(train_df["gen"], train_df["fit"], label="Trained Agent", s=5, alpha=0.33)
    if baseline is not None:
        plt.axhline(y=0, color='r', linestyle='--', label="Baseline")
    plt.title("Fitness per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()

    plt.subplot(2, 4, 2)
    plt.scatter(train_df["gen"], COLLISIONS* (train_df["collisions"]-baseline["collisions"]) / baseline["collisions"], color='g', s=5, alpha=0.33)
    if baseline is not None:
        plt.axhline(y=0, color='r', linestyle='--', label="Baseline")
    plt.title("Collisions per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Collisions")
    plt.legend()

    plt.subplot(2, 4, 3)
    plt.scatter(train_df["gen"], WAIT_TIME*(train_df["waitingTime"]-baseline["waitingTime"]) / baseline["waitingTime"], color='orange', s=5, alpha=0.33)
    if baseline is not None:
        plt.axhline(y=0, color='r', linestyle='--', label="Baseline")
    plt.title("Waiting Time per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Waiting Time")
    plt.legend()

    plt.subplot(2, 4, 4)
    plt.scatter(train_df["gen"], EBRAKES*(train_df["ebrakes"]-baseline["ebrakes"]) / baseline["ebrakes"], color='yellow', s=5, alpha=0.33)
    if baseline is not None:
        plt.axhline(y=0, color='r', linestyle='--', label="Baseline")
    plt.title("Emergency Braking per Generation")
    plt.xlabel("Generation")
    plt.ylabel("E-Brakes")
    plt.legend()

    plt.subplot(2, 4, 5)
    plt.scatter(train_df["gen"], JAM*(train_df["jams"]-baseline["jams"]) / baseline["jams"], color='purple', s=5, alpha=0.33)
    if baseline is not None:
        plt.axhline(y=0, color='r', linestyle='--', label="Baseline")
    plt.title("Teleports (Jams) per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Jams")
    plt.legend()

    plt.subplot(2, 4, 6)
    plt.scatter(train_df["gen"], AVG_TRIP_TIME * (train_df["totalTravelTime"] / train_df["tripCount"]-avgTravelTime) / avgTravelTime, color='blue', s=5, alpha=0.33)
    if avgTravelTime is not None:
        plt.axhline(y=0, color='r', linestyle='--', label="Baseline")
    plt.title("Avg Travel Time per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Avg Travel Time")
    plt.legend()

    plt.subplot(2, 4, 7)
    plt.scatter(train_df["gen"], TRIP_COUNT*(train_df["tripCount"]-baseline["tripCount"]) / baseline["tripCount"], color='cyan', s=5, alpha=0.33)
    if baseline is not None:
        plt.axhline(y=0, color='r', linestyle='--', label="Baseline")
    plt.title("Trip Count per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Trip Count")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Run the script with whichever mode was chosen
if __name__ == '__main__':
    if args.baseline:
        run_baseline()
    elif args.train:
        train()
    elif args.plot:
        plot_metrics()
        plot_weights()
    else:
        print("Please provide an argument: --train, --plot, or --baseline")