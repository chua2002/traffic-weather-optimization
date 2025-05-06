#! /bin/bash


sumo_home="C:\Program Files (x86)\Eclipse\Sumo"

# generate routes
#python "$sumo_home\tools\findAllRoutes.py" -n osm.net.xml.gz -o new.rou.xml

# check and fix routes
python "$sumo_home\tools\route\routecheck.py" -f osm.passenger.trips.xml -v

python "$sumo_home\tools\randomTrips.py" -n osm.net.xml.gz --fringe-factor 5 --insertion-density 12 -o osm.passenger.trips.xml -r osm.passenger.rou.xml -b 0 -e 3600 --trip-attributes "departLane=\"best\"" --fringe-start-attributes "departSpeed=\"max\"" --validate --remove-loops --via-edge-types highway.motorway,highway.motorway_link,highway.trunk_link,highway.primary_link,highway.secondary_link,highway.tertiary_link --vehicle-class passenger --vclass passenger --prefix veh --min-distance 300 --min-distance.fringe 10 --allow-fringe.min-length 1000 --lanes


# create route files and sumocfg for each condition

declare -a conditions=(
[0]=osm_clear_dry
[1]=osm_light_rain
[2]=osm_heavy_rain
[3]=osm_light_snow
[4]=osm_heavy_snow
[5]=osm_fog
)
declare -a cond_properties=(
[0]='<vType id="veh_passenger" vClass="passenger" accel="2.5" decel="3.5" emergencyDecel="4.5" tau="1.0" sigma="0.5" minGap="2.5" startupDelay="0.5" actionStepLength="1.0" carFollowModel="Krauss"/>'
[1]='<vType id="veh_passenger" vClass="passenger" accel="2.2" decel="3.0" emergencyDecel="4.0" tau="0.95" sigma="0.6" minGap="3.0" startupDelay="1.0" actionStepLength="1.2" carFollowModel="Krauss"/>'
[2]='<vType id="veh_passenger" vClass="passenger" accel="1.7" decel="2.5" emergencyDecel="3.5" tau=".9" sigma="0.7" minGap="3.5" startupDelay="1.3" actionStepLength="1.5" carFollowModel="Krauss"/>'
[3]='<vType id="veh_passenger" vClass="passenger" accel="1.3" decel="2.5" emergencyDecel="3.0" tau=".9" sigma="0.7" minGap="3.5" startupDelay="1.3" actionStepLength="1.5" carFollowModel="Krauss"/>'
[4]='<vType id="veh_passenger" vClass="passenger" accel="1.0" decel="2.0" emergencyDecel="2.5" tau=".85" sigma="0.8" minGap="4.5" startupDelay="1.7" actionStepLength="2.0" carFollowModel="Krauss"/>'
[5]='<vType id="veh_passenger" vClass="passenger" accel="1.7" decel="2.5" emergencyDecel="3.5" tau=".85" sigma="0.8" minGap="4.5" startupDelay="1.7" actionStepLength="2.0" carFollowModel="Krauss"/>'
)
for i in {0..5}; do
    cp osm.passenger.trips.xml "${conditions[i]}.passenger.trips.xml"
	
	echo     sed -i 's#<vType id="veh_passenger" vClass="passenger"/>#'"${cond_properties[i]}"'#' "${conditions[i]}.passenger.trips.xml"
    sed -i 's#<vType id="veh_passenger" vClass="passenger"/>#'"${cond_properties[i]}"'#' "${conditions[i]}.passenger.trips.xml"

    cp osm.sumocfg "${conditions[i]}.sumocfg"
    sed -i "s#osm.passenger.trips.xml#${conditions[i]}.passenger.trips.xml#" "${conditions[i]}.sumocfg"
done
