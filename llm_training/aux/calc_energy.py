import csv
import argparse
from datetime import datetime

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--powerfile', type=str, help='path to power data')
args = parser.parse_args()

# Initialize data storage
data = {}
energy = []

# Read the CSV file
with open(args.powerfile, mode='r') as file:
    reader = csv.reader(file)
    header = next(reader)  # Skip the header
    counter = 0
    for row in reader:
        index = row[0]
        try:
            timestamp = datetime.strptime(row[1].strip(), "%Y/%m/%d %H:%M:%S.%f")
            power = float(row[3].replace(' W', '').strip())
        except ValueError:
            counter=counter+1 
            pass

        if index not in data:
            data[index] = []

        data[index].append((timestamp, power))

# Calculate energy consumption per group
for index, records in data.items():
    total_energy = 0.0
    for i in range(1, len(records)):
        time_diff = (records[i][0] - records[i-1][0]).total_seconds()  # Time difference in seconds
        power = records[i-1][1]  # Power in watts
        total_energy += power * time_diff  # Energy in Joules
    energy.append(total_energy / 3600)  # Convert energy to Wh and store it
# Output the energy per GPU
print(f"Energy-per-GPU-list integrated(Wh): {energy}")