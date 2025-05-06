#!/bin/bash
export PYTHONPATH="/c/Program Files (x86)/Eclipse/Sumo/tools"

sims=("dadeville_v2") # ("gordo" "dadeville_v2" "stark_reduced_v2"
w_conditions=("osm_fog" "osm_clear_dry" "osm_heavy_rain" "osm_heavy_snow" "osm_light_rain" "osm_light_snow")
set -e

for i in "${sims[@]}"; do
    for j in "${w_conditions[@]}"; do
	(
        echo "Running: scenario=$i, weather=$j"
#
        python genetic.py --scenario "$i" --weather "$j" --baseline > /dev/null
        python genetic.py --scenario "$i" --weather "$j" --train > /dev/null


        mv "results/${i}_${j}/baseline_data.csv" "compiled_results/${i}_${j}_baseline.csv"
        mv "results/${i}_${j}/metric_data.csv" "compiled_results/${i}_${j}_metric.csv"
        mv "results/${i}_${j}/gene_data.csv" "compiled_results/${i}_${j}_gene.csv"

        echo "Finished: scenario=$i, weather=$j"
        echo "-----------------------------------"
	) &
    done
    wait
done