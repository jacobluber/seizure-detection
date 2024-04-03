#!/bin/bash

# Define the output file
output_file="DGX_Power_1.csv"

# Run the Python inference script in the background
python3 Inference.py &

# Store the background process ID of the Python script
python_pid=$!


# Get start time
start_time=$(date +%s.%N)

# Loop to collect data
while true; do
    # Check if the Python script has finished
    if ! ps -p $python_pid > /dev/null; then
        echo "Inference.py finished, stopping data collection."
        break
    fi

    # Get elapsed time since start in seconds with fractions
    current_time=$(date +%s.%N)
    elapsed_time=$(echo "$current_time - $start_time" | bc)
    
    # Collect measurement data				 for gpu watts	   for gpu watts
    measurement=$(nvidia-smi | grep "N/A" | sed 's/W.*/W/g' | sed 's/.*P0//g' | head -1)
    
    # Append data to the output file
    echo "$elapsed_time,$measurement" >> "$output_file"
	
    sleep 0.001  # Sleep for 1 millisecond
done

echo "Data collection complete."
