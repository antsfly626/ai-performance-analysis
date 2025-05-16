import os
import pandas as pd
import matplotlib.pyplot as plt
import re

# set directory and select model
directory = "./data/inference_edge/closed"
model_name = "gptj-99"  # example

# get version number from filename
def get_version(filename):
    match = re.search(r'v(\d+\.\d+)', filename)
    if match:
        return float(match.group(1))
    return 0.0  # if no version found

# load files and organize by version
version_data = {}

for file in os.listdir(directory):
    if file.endswith(".csv"):
        version = get_version(file)
        if version > 0:  # Only process files with valid version numbers
            df = pd.read_csv(os.path.join(directory, file), on_bad_lines='skip')
            version_data[version] = df

# Sort versions
sorted_versions = sorted(version_data.keys())

# updated plot function
def plot_all_versions(model_name):
    for version in sorted_versions:
        df = version_data[version]
        
        # filter for specific model
        filtered = df[df["Model"].str.lower() == model_name.lower()]
        
        if filtered.empty:
            print(f"No entries for {model_name} in version {version}")
            continue
        
        # Get latency and throughput
        latency_data = filtered[filtered["Units1"].str.contains("Latency", case=False, na=False)]
        throughput_data = filtered[~filtered["Units1"].str.contains("Latency", case=False, na=False)]
        
        # Merge latency and throughput on System Name
        merged = pd.merge(latency_data, throughput_data, 
                          on="System Name (click + for details)", 
                          suffixes=('_latency', '_throughput'))
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for idx, row in merged.iterrows():
            system_name = row["System Name (click + for details)"]
            latency = row["Result_latency"]
            throughput = row["Result_throughput"]
            
            ax.scatter(throughput, latency, label=system_name)
            ax.annotate(system_name, (throughput, latency), textcoords="offset points", xytext=(5,5), ha='left', fontsize=8)
        
        ax.set_xlabel('Throughput (samples/sec)')
        ax.set_ylabel('Latency (ms)')
        ax.set_title(f"Version {version}: Latency vs Throughput for {model_name}")
        ax.grid(True)
        
        plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

        print(f"Plotted version {version}")

plot_all_versions(model_name)
