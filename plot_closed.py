import os
import pandas as pd
import matplotlib.pyplot as plt
import re

directory = "./data/inference_edge/closed"
#model_name = "gptj-99"
model_name = "BERT-99.0"
#model_name = "3D U-Net-99.9"
def get_version(filename):
    match = re.search(r'v(\d+\.\d+)', filename)
    return float(match.group(1)) if match else None

# load and organize data by version
version_data = {}
for file in os.listdir(directory):
    if file.endswith(".csv"):
        version = get_version(file)
        if version:
            df = pd.read_csv(os.path.join(directory, file), on_bad_lines='skip')
            version_data[version] = df

# sort versions
sorted_versions = sorted(version_data.keys())

# make dictionary list of throughput, latency pairs across versions
system_points = {}

for version in sorted_versions:
    df = version_data[version]
    filtered = df[df["Model"].str.lower() == model_name.lower()]

    if filtered.empty:
        continue

    latency_df = filtered[filtered["Units1"].str.contains("Latency", case=False, na=False)]
    throughput_df = filtered[~filtered["Units1"].str.contains("Latency", case=False, na=False)]

    merged = pd.merge(latency_df, throughput_df, 
                      on="System Name (click + for details)", 
                      suffixes=('_latency', '_throughput'))

    for _, row in merged.iterrows():
        system = row["System Name (click + for details)"]
        latency = row["Result_latency"]
        throughput = row["Result_throughput"]

        if system not in system_points:
            system_points[system] = []

        system_points[system].append((throughput, latency, version))

# Plotting
fig, ax = plt.subplots(figsize=(10, 7))

for system, points in system_points.items():
    # sort system's points by version to connect them in version order
    points.sort(key=lambda x: x[2])
    x = [p[0] for p in points]  # throughput
    y = [p[1] for p in points]  # latency
    ax.plot(x, y, marker='o', label=system)

ax.set_xlabel("Throughput (samples/sec)")
ax.set_ylabel("Latency (ms)")
ax.set_title(f"{model_name.upper()}: Latency vs Throughput Across Versions")
ax.grid(True)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.show()
