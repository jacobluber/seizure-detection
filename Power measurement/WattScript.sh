#!/bin/bash

# Define the output file
output_file="measurements.csv"

# Run the Python inference script in the background
python3 Inference.py &

# Loop to collect data
for ((i=1; i<=1000; i++)); do
    nvidia-smi | grep "N/A" | sed 's/W.*/W/g' | sed 's/.*P0//g' | head -1 >> "$output_file"
    sleep 0.001  # Sleep for 1 millisecond
done

# Wait for the Python inference script to finish
wait

echo "Data collection complete."