# Traffic Weather Optimization
## Goal:
Find optimal speeds for traffic safety and flow in adverse weather conditions. Gain insight from comparing optimized traffic networks in different conditions.
## Method:

### Use data about driver behaviour to set vehicle parameters for different weather conditions in real world locations.
| Weather Condition | Acceleration (m/s²) | Deceleration (m/s²) | Emergency Deceleration (m/s²) | Reaction Time (tau, s) | Driver Imperfection (sigma) | Minimum Gap (minGap, m) | Startup Delay (s) | Action Step Length (s) |
| ----------------- | -------------------- | -------------------- | ------------------------------ | ----------------------- | ---------------------------- | ------------------------ | ------------------ | ----------------------- |
| Clear/Dry         | 2.5                  | 3                    | 4.5                            | 1                       | 0.5                          | 2.5                      | 0.5                | 1                       |
| Light Rain        | 2                    | 2.5                  | 3.5                            | 0.95                    | 0.6                          | 3                        | 1.0                | 1.2                     |
| Heavy Rain        | 1.5                  | 2                    | 3                              | 0.9                     | 0.7                          | 3.5                      | 1.2                | 1.5                     |
| Light Snow        | 1                    | 2                    | 2.5                            | 0.9                     | 0.7                          | 3.5                      | 1.2                | 1.5                     |
| Heavy Snow        | 0.8                  | 1.5                  | 2                              | 0.85                    | 0.8                          | 4                        | 1.5                | 2                       |
| Fog               | 1.5                  | 2                    | 3                              | 0.85                    | 0.8                          | 4                        | 1.5                | 2                       |

### Employ a genetic optimization of road speed limits to improve driver safety metrics.

Each road's is a speed limit is a gene. Individuals in then population are comprised of genes for each road.
Each individual is run through 1000 SUMO traffic simulator simulation steps, then has its metrics put though a fitness function.

Parents for the next generation are selected roulette wheel style, weighted by their fitness. Genes for the new individual are an extended intermediate recombination of parent genes.

Additionally, there is a chance for random mutation for every gene. This probability decays each generation.

Elitism is employed to prevent population fitness degredation. Put simply, the highest fitness individual preserved into the next generation.

We recommend running at least 100 generations to see convergence.

### Run the optimization in a variety of scenenarios for more generalizable insights.

Included in the repository are 5 scenarios: Gordo, McComb, Dadeville, Starkville, and Columbus Air Force Base (CAFB).
Note that run times for scenarios vary, with Starkville being the greatest, 20 hours,
and less than 1 hour for Gordo and Dadeville (ryzen 5 5600X, 100 generations).

### Compare distribution of optimized speed limits accross weather conditions and locations.

#### process_data.py 
Functions within file to run and generate graphs
| function | use |
| --- | --- |
| plot_compare_metrics | compare optimized weather metrics for a single scenario |
| plot_metrics | plot metrics over the course of training for a single weather condition |
| plot_speeds | for a given scenario, plot distribution of speed limits

#### plot_net_speeds_custom.py
Altered built-in SUMO script. Displays traffic network with color-coded speed limits

```
python plot_net_speeds_custom.py -n <network_path> --minV 0 --maxV 100 --scenario <scenario> --weather <weather_condition> --colormap "hot"
```

speed_net.sh is a helper script to make this easier.
