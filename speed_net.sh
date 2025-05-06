#!/bin/bash

scenario="dadeville_v2"
weather="osm_heavy_snow"
net_path="${scenario}/osm.net.xml.gz"


python plot_net_speeds_custom.py -n $net_path --minV 0 --maxV 100 --scenario $scenario --weather $weather   --colormap "hot"