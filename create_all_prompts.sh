#!/bin/zsh
#
# Iterate over each file and run the Python script
for i in 1 2 3; do
    python create_tables.py data/navigation_scenario"$i"_edge.txt;
    python generate_prompt.py data/navigation_scenario"$i"_edge.txt-filtered_table.txt;
    python generate_prompt.py data/navigation_scenario"$i"_edge.txt-raw_obs_table.txt;
done
