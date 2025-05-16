import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from collections import defaultdict

# Configuration settings
MODEL_NAME = "ResNet"  # Model to analyze
SCENARIO = "all"  # Options: "all", "SingleStream", "Offline", "MultiStream"
OUTPUT_DIR = "power_latency_plots"
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

# Check the unique models in the dataset
unique_models = combined_df['Model'].dropna().unique()
print(f"\nAvailable models: {unique_models}")

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

# Check available accelerators for this model
accelerators = model_data['Accelerator'].dropna().unique()
print(f"\nAccelerators for {MODEL_NAME}: {accelerators}")

# Function to get power and latency data for a specific scenario and accelerator
def get_scenario_data(model_df, scenario, accelerator=None):
    # Filter by scenario
    scenario_data = model_df[model_df["Scenario1"] == scenario]
    
    # Filter by accelerator if specified
    if accelerator and accelerator != "all":
        scenario_data = scenario_data[scenario_data["Accelerator"] == accelerator]
    
    # Group by system and version to collect paired measurements
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
                acc = latency_row["Accelerator"] if pd.notna(latency_row["Accelerator"]) else "CPU"
                
                # Get number of accelerators if available
                num_acc = 1  # Default to 1
                if "# of Accelerators" in latency_row and pd.notna(latency_row["# of Accelerators"]):
                    try:
                        num_acc = float(latency_row["# of Accelerators"])
                        if num_acc <= 0:
                            num_acc = 1
                    except:
                        pass
                
                # Store the data point
                result_data.append({
                    "System": system,
                    "Version": version,
                    "Accelerator": acc,
                    "Latency": float(latency_row["Result"]),
                    "Power": float(power_row["Result"]),
                    "PowerPerAccelerator": float(power_row["Result"]) / num_acc,
                    "Units_Latency": latency_row["Units1"],
                    "Units_Power": power_row["Units1"],
                    "NumAccelerators": num_acc
                })
    
    return pd.DataFrame(result_data) if result_data else None

# Process data for each scenario and accelerator
scenario_accelerator_data = {}

for scenario in scenarios:
    print(f"\nProcessing {scenario} scenario")
    scenario_data = defaultdict(list)
    
    # Process data for each accelerator separately
    for acc in list(accelerators) + [None]:  # None to include CPUs without explicit accelerator value
        acc_name = acc if pd.notna(acc) else "CPU/Unknown"
        df = get_scenario_data(model_data, scenario, acc)
        
        if df is not None and not df.empty:
            print(f"  Found {len(df)} data points for {acc_name}")
            scenario_data[acc_name] = df
    
    # Store results for plotting
    scenario_accelerator_data[scenario] = scenario_data

# Function to create a plot for a specific scenario and accelerator
def create_plot(scenario, accelerator, df, output_dir):
    if df.empty:
        print(f"No data for {accelerator} in {scenario}")
        return
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by version for connecting lines
    df = df.sort_values("Version")
    
    # Create scatter plot
    scatter = ax.scatter(
        df["Latency"], 
        df["PowerPerAccelerator"], 
        c=df["Version"], 
        cmap='viridis', 
        s=100
    )
    
    # Connect points in version order if more than one
    if len(df) > 1:
        ax.plot(df["Latency"], df["PowerPerAccelerator"], 'k--', alpha=0.5)
    
    # Add version labels
    for i, row in df.iterrows():
        ax.annotate(
            f"v{row['Version']}", 
            (row["Latency"], row["PowerPerAccelerator"]), 
            textcoords="offset points", 
            xytext=(0, 10), 
            ha='center'
        )
    
    # Add system labels
    for i, row in df.iterrows():
        ax.annotate(
            f"{row['System'][:20]}...", 
            (row["Latency"], row["PowerPerAccelerator"]), 
            textcoords="offset points", 
            xytext=(0, -15), 
            ha='center', 
            fontsize=8
        )
    
    # Set plot labels and title
    ax.set_title(f"{MODEL_NAME} on {accelerator}: Power vs Latency ({scenario})")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Power per Accelerator (W)")
    ax.grid(True, alpha=0.3)
    
    # Add colorbar to show version scale
    cbar = plt.colorbar(scatter)
    cbar.set_label('Version')
    
    # Save the plot
    safe_acc_name = str(accelerator).replace(' ', '_').replace('/', '_').replace(',', '')
    safe_scenario = scenario.replace(' ', '_')
    filename = os.path.join(output_dir, f"{MODEL_NAME}_{safe_acc_name}_{safe_scenario}.png")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close()

# Function to create a combined plot with all accelerators for a scenario
def create_combined_plot(scenario, accelerator_dfs, output_dir):
    if not accelerator_dfs:
        print(f"No data for any accelerator in {scenario}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use a distinct color for each accelerator
    accelerators = list(accelerator_dfs.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(accelerators)))
    
    for i, (acc_name, df) in enumerate(accelerator_dfs.items()):
        if df.empty:
            continue
        
        # Sort by version
        df = df.sort_values("Version")
        
        # Plot with unique color
        ax.scatter(df["Latency"], df["PowerPerAccelerator"], s=100, color=colors[i], label=acc_name)
        
        # Connect points if more than one
        if len(df) > 1:
            ax.plot(df["Latency"], df["PowerPerAccelerator"], '-', color=colors[i], alpha=0.7)
        
        # Add version labels
        for j, row in df.iterrows():
            ax.annotate(
                f"v{row['Version']}", 
                (row["Latency"], row["PowerPerAccelerator"]), 
                textcoords="offset points", 
                xytext=(0, 10), 
                ha='center', 
                color=colors[i]
            )
    
    # Set plot labels, title, and legend
    ax.set_title(f"{MODEL_NAME}: Power vs Latency - {scenario} Scenario")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Power per Accelerator (W)")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save the plot
    safe_scenario = scenario.replace(' ', '_')
    filename = os.path.join(output_dir, f"{MODEL_NAME}_all_accelerators_{safe_scenario}.png")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved combined plot: {filename}")
    plt.close()

# Create plots for each scenario and accelerator
for scenario, accelerator_data in scenario_accelerator_data.items():
    print(f"\nCreating plots for {scenario} scenario")
    
    # Create individual plots for each accelerator
    for acc_name, df in accelerator_data.items():
        create_plot(scenario, acc_name, df, OUTPUT_DIR)
    
    # Create combined plot for this scenario
    create_combined_plot(scenario, accelerator_data, OUTPUT_DIR)

print("\nAnalysis completed!")