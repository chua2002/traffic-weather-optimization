
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

LOG_FOLDER = "compiled_results"

# sims=("gordo", "dadeville_v2")

sims=[ "dadeville_v2"]
w_conditions=("osm_clear_dry","osm_light_snow", "osm_heavy_snow", "osm_light_rain", "osm_heavy_rain", "osm_fog")
w_colors=("goldenrod", "magenta", "purple", "dodgerblue", "navy", "slategray")
best_speeds = {}
best_metrics = {}


for s in sims:
    best_metrics[s] = {}
    best_speeds[s] = {}
    for w in w_conditions:
        stats = np.array(pd.read_csv(f'{LOG_FOLDER}/{s}_{w}_metric.csv', usecols=[2, 3, 4, 5, 6, 7, 8], header=None, dtype=float))
        best_idx = int(np.argmax(stats[:, 0]))

        best_metrics[s][w] = {"jams":stats[best_idx,1], "ebrakes":stats[best_idx,2], "collisions":stats[best_idx,3],
                              "totalTravelTime":stats[best_idx,4], "waitingTime":stats[best_idx,5],
                              "tripCount":stats[best_idx,6]}

        speeds = np.array(pd.read_csv(f'{LOG_FOLDER}/{s}_{w}_gene.csv', skiprows=best_idx, nrows=1, header=None, dtype=float))
        best_speeds[s][w] = speeds[0]




def plot_speeds(location):

    bins = range(0,101,10)
    plt.hist([x for x in best_speeds[location].values()], bins, label=[x for x in best_speeds[location]], color=w_colors)
    plt.xlabel('Speed Limit (mph)')
    plt.ylabel('Frequency')
    plt.title(f'{location} speeds in different weather conditions')
    plt.legend(loc='upper right') # Adjust the location as needed
    plt.xticks(ticks=bins)


    means = [np.mean(x) for x in best_speeds[location].values()]
    for i in range(len(w_conditions)):
        print(w_conditions[i], means[i])
        plt.axvline(x=means[i], linestyle="--", color=w_colors[i], alpha=1)

    plt.show()

def plot_metrics(SCENARIO, WEATHER):
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
        plt.axhline(y=baseline["collisions"]-1, color='r', linestyle='--', label="Baseline")
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

def plot_compare_metrics(scenario):

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 3, 1)
    plt.bar(w_conditions, [best_metrics[scenario][w]["collisions"] for w in w_conditions], color=w_colors)
    plt.xticks(rotation='vertical')
    plt.title("Collisions Across Weather Conditions")
    plt.xlabel("Weather condition")
    plt.ylabel("Collisions")

    plt.subplot(2, 3, 2)
    plt.bar(w_conditions, [best_metrics[scenario][w]["waitingTime"] for w in w_conditions], color=w_colors)
    plt.xticks(rotation='vertical')
    plt.title("Waiting Time Across Weather Conditions")
    plt.xlabel("Weather Condition")
    plt.ylabel("waiting Time")

    plt.subplot(2, 3, 3)
    plt.bar(w_conditions, [best_metrics[scenario][w]["ebrakes"] for w in w_conditions], color=w_colors)
    plt.xticks(rotation='vertical')
    plt.title("E-Brakes Across Weather Conditions")
    plt.xlabel("Weather condition")
    plt.ylabel("E-Brakes")

    plt.subplot(2, 3, 4)
    plt.bar(w_conditions, [best_metrics[scenario][w]["jams"] for w in w_conditions], color=w_colors)
    plt.xticks(rotation='vertical')
    plt.title("Jams Across Weather Conditions")
    plt.xlabel("Weather condition")
    plt.ylabel("Jams")

    plt.subplot(2, 3, 5)
    plt.bar(w_conditions, [best_metrics[scenario][w]["totalTravelTime"]/best_metrics[scenario][w]["tripCount"] for w in w_conditions], color=w_colors)
    plt.xticks(rotation='vertical')
    plt.title("Average Trip Time Across Weather Conditions")
    plt.xlabel("Weather condition")
    plt.ylabel("Average Trip Time")

    plt.subplot(2, 3, 6)
    plt.bar(w_conditions, [best_metrics[scenario][w]["tripCount"] for w in w_conditions], color=w_colors)
    plt.xticks(rotation='vertical')
    plt.title("Trip Counts Across Weather Conditions")
    plt.xlabel("Weather condition")
    plt.ylabel("Trip Count")

    plt.tight_layout()
    plt.show()



# plot_speeds("gordo")
# plot_metrics("gordo", "osm_clear_dry")
# plot_metrics("gordo", "osm_heavy_snow")
#
# plot_metrics("gordo", "osm_fog")
#
# plot_compare_metrics("gordo")



plot_speeds("dadeville_v2")
plot_metrics("dadeville_v2", "osm_clear_dry")
plot_metrics("dadeville_v2", "osm_light_rain")
plot_metrics("dadeville_v2", "osm_fog")

plot_compare_metrics("dadeville_v2")
