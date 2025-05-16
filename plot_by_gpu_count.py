import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from collections import defaultdict

# Configuration settings
MODEL_NAME = "ResNet"  # Model to analyze
SCENARIO = "all"  # Options: "all", "SingleStream", "Offline", "MultiStream"
OUTPUT_DIR = "accelerator_count_plots"
DATA_DIR = "./data/inference_edge/open-power"

def get_version(filename):
    match = re.search(r'v(\d+\.\d+)', filename)
    return float(match.group(1)) if match else None

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load and combine all CSV files
all_data = []
files_found = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
print(f"Found {len(files_found)} CSV files in {DATA_DIR}")

for file in files_found:
    file_path = os.path.join(DATA_DIR, file)
    version = get_version(file)
    if version:
        try:
            df = pd.read_csv(file_path, on_bad_lines='skip')
            df['Version'] = version
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")

if not all_data:
    print("No data could be loaded!")
    exit()

# Combine all data into a single dataframe
combined_df = pd.concat(all_data, ignore_index=True)
print("\nColumns in the dataset:", combined_df.columns.tolist())

# Filter for the model we want
model_data = combined_df[combined_df["Model"].str.contains(MODEL_NAME, case=False, na=False)]
print(f"\nFound {len(model_data)} rows for {MODEL_NAME} across all versions")

# Print sample row to understand structure
if not model_data.empty:
    print("\nSample row:")
    print(model_data.iloc[0].to_string())

# Filter for the specified scenario if needed
if SCENARIO != "all":
    model_data = model_data[model_data["Scenario1"] == SCENARIO]
    print(f"Filtered for {SCENARIO} scenario: {len(model_data)} rows")

# Get unique scenarios for processing
scenarios = model_data["Scenario1"].unique() if SCENARIO == "all" else [SCENARIO]
print(f"Analyzing scenarios: {scenarios}")

# Process the "# of Accelerators" column
model_data['AcceleratorCount'] = pd.to_numeric(model_data['# of Accelerators'], errors='coerce')
model_data['AcceleratorCount'].fillna(1, inplace=True)  # Default to 1 if missing
model_data['AcceleratorCount'] = model_data['AcceleratorCount'].astype(int)

# Check distribution of accelerator counts
acc_counts = model_data['AcceleratorCount'].value_counts().sort_index()
print("\nAccelerator count distribution:")
for count, freq in acc_counts.items():
    print(f"{count} accelerator(s): {freq} data points")

# Function to get power and latency data for a specific scenario, grouped by accelerator count
def get_scenario_data_by_count(model_df, scenario):
    # Filter by scenario
    scenario_data = model_df[model_df["Scenario1"] == scenario]
    
    # Group by system, version, and accelerator count to collect paired measurements
    systems = scenario_data["System Name (click + for details)"].unique()
    result_data = []
    
    for system in systems:
        system_data = scenario_data[scenario_data["System Name (click + for details)"] == system]
        
        for version in system_data["Version"].unique():
            version_data = system_data[system_data["Version"] == version]
            
            # Find latency and power data
            latency_row = None
            power_row = None
            
            # Look for latency values
            latency_candidates = version_data[version_data["Units1"].str.contains("latency", case=False, na=False)]
            if not latency_candidates.empty:
                latency_row = latency_candidates.iloc[0]
            
            # Look for power values (System Watts)
            power_candidates = version_data[version_data["Units1"].str.contains("Watts", case=False, na=False)]
            if not power_candidates.empty:
                power_row = power_candidates.iloc[0]
            
            # Alternative: Look for energy per stream
            if power_row is None:
                energy_candidates = version_data[version_data["Units1"].str.contains("energy", case=False, na=False)]
                if not energy_candidates.empty:
                    power_row = energy_candidates.iloc[0]
            
            # If we have both latency and power data
            if latency_row is not None and power_row is not None:
                # Get the accelerator info
                acc_type = latency_row["Accelerator"] if pd.notna(latency_row["Accelerator"]) else "CPU"
                
                # Get number of accelerators
                if "AcceleratorCount" in latency_row:
                    acc_count = latency_row["AcceleratorCount"]
                else:
                    acc_count = 1  # Default to 1
                
                # Store the data point
                result_data.append({
                    "System": system,
                    "Version": version,
                    "AcceleratorType": acc_type,
                    "AcceleratorCount": acc_count,
                    "Latency": float(latency_row["Result"]),
                    "Power": float(power_row["Result"]),
                    "PowerPerAccelerator": float(power_row["Result"]) / acc_count,
                    "Units_Latency": latency_row["Units1"],
                    "Units_Power": power_row["Units1"]
                })
    
    return pd.DataFrame(result_data) if result_data else None

# Process data for each scenario
scenario_data_by_count = {}

for scenario in scenarios:
    print(f"\nProcessing {scenario} scenario")
    df = get_scenario_data_by_count(model_data, scenario)
    
    if df is not None and not df.empty:
        print(f"  Found {len(df)} data points")
        # Group by accelerator count
        counts = df["AcceleratorCount"].unique()
        print(f"  Accelerator counts: {counts}")
        scenario_data_by_count[scenario] = df
    else:
        print(f"  No paired latency and power data found for {scenario}")

# Function to create plots by accelerator count
def create_acc_count_plot(scenario, df, output_dir):
    if df.empty:
        print(f"No data for {scenario}")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Group by accelerator count
    acc_counts = sorted(df["AcceleratorCount"].unique())
    
    # Create a colormap for accelerator counts
    colors = plt.cm.tab10(np.linspace(0, 1, len(acc_counts)))
    
    # Plot for each accelerator count
    for i, count in enumerate(acc_counts):
        count_df = df[df["AcceleratorCount"] == count]
        count_df = count_df.sort_values("Version")
        
        # Plot with unique color
        scatter = ax.scatter(
            count_df["Latency"], 
            count_df["Power"],  # Using total power
            s=100, 
            color=colors[i], 
            label=f"{count} Accelerator(s)"
        )
        
        # Add labels for system and version
        for _, row in count_df.iterrows():
            ax.annotate(
                f"v{row['Version']}\n{row['System'][:10]}...", 
                (row["Latency"], row["Power"]), 
                textcoords="offset points", 
                xytext=(0, 10), 
                ha='center', 
                fontsize=8,
                color=colors[i]
            )
    
    # Set plot labels, title, and legend
    ax.set_title(f"{MODEL_NAME}: Total Power vs Latency by Accelerator Count - {scenario}")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Total System Power (W)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Accelerator Count")
    
    # Save the plot
    safe_scenario = scenario.replace(' ', '_')
    filename = os.path.join(output_dir, f"{MODEL_NAME}_acc_count_power_latency_{safe_scenario}.png")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close()
    
    # Create a second plot with Power Per Accelerator
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot for each accelerator count
    for i, count in enumerate(acc_counts):
        count_df = df[df["AcceleratorCount"] == count]
        count_df = count_df.sort_values("Version")
        
        # Plot with unique color
        scatter = ax.scatter(
            count_df["Latency"], 
            count_df["PowerPerAccelerator"],  # Using power per accelerator
            s=100, 
            color=colors[i], 
            label=f"{count} Accelerator(s)"
        )
        
        # Add labels for system and version
        for _, row in count_df.iterrows():
            ax.annotate(
                f"v{row['Version']}\n{row['System'][:10]}...", 
                (row["Latency"], row["PowerPerAccelerator"]), 
                textcoords="offset points", 
                xytext=(0, 10), 
                ha='center', 
                fontsize=8,
                color=colors[i]
            )
    
    # Set plot labels, title, and legend
    ax.set_title(f"{MODEL_NAME}: Power Per Accelerator vs Latency - {scenario}")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Power Per Accelerator (W)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Accelerator Count")
    
    # Save the plot
    filename = os.path.join(output_dir, f"{MODEL_NAME}_acc_count_power_per_acc_{safe_scenario}.png")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close()

# Create efficiency plot (performance per watt)
def create_efficiency_plot(scenario, df, output_dir):
    if df.empty:
        return
    
    # Calculate performance per watt (1/latency per watt)
    df['PerformancePerWatt'] = 1000.0 / (df['Latency'] * df['Power'])  # Higher is better
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Group by accelerator count
    acc_counts = sorted(df["AcceleratorCount"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(acc_counts)))
    
    # Plot for each accelerator count
    for i, count in enumerate(acc_counts):
        count_df = df[df["AcceleratorCount"] == count]
        count_df = count_df.sort_values("Version")
        
        # Skip if no data
        if count_df.empty:
            continue
        
        # Plot with unique color
        bar_positions = np.arange(len(count_df)) + i*0.2
        bars = ax.bar(
            bar_positions,
            count_df["PerformancePerWatt"],
            width=0.2,
            color=colors[i],
            label=f"{count} Accelerator(s)"
        )
        
        # Add labels for system
        for j, row in count_df.iterrows():
            ax.text(
                bar_positions[j],
                row["PerformancePerWatt"] / 2,
                f"{row['System'][:10]}...\nv{row['Version']}",
                ha='center',
                va='center',
                fontsize=8,
                rotation=90,
                color='white' if row["PerformancePerWatt"] > 0.15 else 'black'
            )
    
    # Set plot labels, title, and legend
    ax.set_title(f"{MODEL_NAME}: Performance per Watt - {scenario}")
    ax.set_ylabel("Performance per Watt (1/ms/W)")
    ax.set_xticks([])  # Hide x-axis ticks since we're using custom labels
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(title="Accelerator Count")
    
    # Save the plot
    safe_scenario = scenario.replace(' ', '_')
    filename = os.path.join(output_dir, f"{MODEL_NAME}_efficiency_{safe_scenario}.png")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved efficiency plot: {filename}")
    plt.close()

# Create scaling plot (speedup vs accelerator count)
def create_scaling_plot(scenario, df, output_dir):
    if df.empty or len(df["AcceleratorCount"].unique()) <= 1:
        return
    
    # Group by accelerator type to compare scaling within the same hardware family
    acc_types = df["AcceleratorType"].dropna().unique()
    
    for acc_type in acc_types:
        acc_df = df[df["AcceleratorType"] == acc_type]
        
        # Skip if not enough data
        if len(acc_df["AcceleratorCount"].unique()) <= 1:
            continue
            
        # Get baseline latency (single accelerator)
        single_acc_data = acc_df[acc_df["AcceleratorCount"] == 1]
        if single_acc_data.empty:
            continue  # Skip if no single accelerator baseline
            
        baseline_latency = single_acc_data["Latency"].mean()
        
        # Calculate speedup
        acc_df["Speedup"] = baseline_latency / acc_df["Latency"]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot actual speedup
        counts = sorted(acc_df["AcceleratorCount"].unique())
        speedups = [acc_df[acc_df["AcceleratorCount"] == c]["Speedup"].mean() for c in counts]
        
        ax.plot(counts, speedups, 'o-', linewidth=2, markersize=10, label="Actual Speedup")
        
        # Plot ideal linear speedup
        ideal_speedups = [c for c in counts]
        ax.plot(counts, ideal_speedups, '--', color='gray', label="Ideal Linear Speedup")
        
        # Add system name annotations
        for count, speedup in zip(counts, speedups):
            systems = acc_df[acc_df["AcceleratorCount"] == count]["System"].unique()
            ax.annotate(
                f"{systems[0][:15]}...",
                (count, speedup),
                textcoords="offset points", 
                xytext=(0, 10), 
                ha='center',
                fontsize=8
            )
        
        # Set plot labels, title, and legend
        ax.set_title(f"{MODEL_NAME}: Scaling Efficiency for {acc_type} - {scenario}")
        ax.set_xlabel("Number of Accelerators")
        ax.set_ylabel("Speedup (relative to 1 accelerator)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set x-axis to show integers
        ax.set_xticks(counts)
        
        # Save the plot
        safe_scenario = scenario.replace(' ', '_')
        safe_acc_type = str(acc_type).replace(' ', '_').replace('/', '_').replace(',', '')
        filename = os.path.join(output_dir, f"{MODEL_NAME}_scaling_{safe_acc_type}_{safe_scenario}.png")
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Saved scaling plot for {acc_type}: {filename}")
        plt.close()

# Create plots for each scenario
for scenario, df in scenario_data_by_count.items():
    print(f"\nCreating plots for {scenario} scenario")
    create_acc_count_plot(scenario, df, OUTPUT_DIR)
    create_efficiency_plot(scenario, df, OUTPUT_DIR)
    create_scaling_plot(scenario, df, OUTPUT_DIR)

print("\nAnalysis completed!")