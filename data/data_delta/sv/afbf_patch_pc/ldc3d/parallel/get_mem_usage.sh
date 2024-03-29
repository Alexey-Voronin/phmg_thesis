#!/bin/bash

# The input file
input_file="job_map.log"
# The output file
output_file="job_map_mem.log"

# Check if the output file already exists and remove it to start fresh
[ -f "$output_file" ] && rm "$output_file"

# Read the input file line by line
while IFS= read -r line; do
  # Extract the job number from the line
  job_number=$(echo "$line" | awk -F' --> ' '{print $2}')
  # Call seff for the job number and capture the output
  seff_output=$(seff "$job_number" | grep Mem)
  # Write the original line and the seff output to the output file
  echo "$line" >> "$output_file"
  echo "$seff_output" >> "$output_file"

done < "$input_file"

