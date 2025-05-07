#!/bin/bash

sims=("dadeville_v2") # "gordo" "stark_reduced_v2" "mccomb" "cafb"
w_conditions=("osm_fog" "osm_clear_dry" "osm_heavy_rain" "osm_heavy_snow" "osm_light_rain" "osm_light_snow")




for i in "${sims[@]}"; do
    for j in "${w_conditions[@]}"; do
	(
        echo "Drawing Speed Net: scenario=$i, weather=$j"
        python plot_net_speeds_custom.py -n "${i}/osm.net.xml.gz" --minV 0 --maxV 100 --scenario $i --weather $j   --colormap "hot"

        mv "${i}_${j}.png" "compiled_results/"

        echo "Finished: scenario=$i, weather=$j"
	) &
    done
    wait
done