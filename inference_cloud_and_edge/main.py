'''
Week 2 Meeting Notes
Download datasets as csv
Group by organization and accelerator
Given accelerator and organization get model
Plot power consumption and latency
Merge closed and closed power
See documentation and see differences between closed/closed power
https://mlcommons.org/benchmarks/inference-edge/
Plot everything
Edge is very small server and cloud is very large access to hardware
Plot when it is edge and cloud
Explain all nuances and differences and plot things
'''

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read the data from the csv files

# cloud
df1 = pd.read_csv('inference_datacenter_closed.csv')
df2 = pd.read_csv('inference_datacenter_closed_power.csv')

# edge
df3 = pd.read_csv('inference_edge_closed.csv')
df4 = pd.read_csv('inference_edge_closed_power.csv')

# merge df1 and df2, adding a new column called Division/Power that will say which in the merged are closed or closed - power
def merge_dataframes(df1, df2):
    # merge the two dataframes
    merged_df = pd.merge(df1, df2, on=['Public ID', 'Organization', 'Availability', 'System Name (click + for details)', '# of Nodes', 'Processor', 'Accelerator', '# of Accelerators', 'Benchmark', 'Model MLC', 'Scenario', 'Units', 'Valid / Invalid', '# of Accelerators', '# of Nodes', 'Accelerator', 'Availability', 'Host Processor Core Count', 'Processor'], how='outer')
    # add a new column called Division/Power that will say which in the merged are closed or closed - power
    merged_df['Division/Power'] = np.where(merged_df['Avg. Result at System Name_x'].isnull(), 'Closed - Power', 'Closed') # avg. result at system name is the column that has the inference time and power consumption
    return merged_df

# call the merge_dataframes function
merged_df1 = merge_dataframes(df1, df2) # cloud

# merge df3 and df4, adding a new column called Division/Power that will say which in the merged are closed or closed - power
def merge_dataframes_edge(df3, df4):
    # merge the two dataframes
    merged_df_edge = pd.merge(df3, df4, on=['System Name (click + for details)', 'Accelerator', '# of Accelerators', 'Benchmark1', 'Model MLC1', 'Scenario1', 'Units1', '# of Accelerators', '# of Nodes', 'Accelerator', 'Host Processor Core Count', 'Processor'], how='outer')
    # add a new column called Division/Power that will say which in the merged are closed or closed - power
    merged_df_edge['Division/Power'] = np.where(merged_df_edge['Avg. Result at System Name_x'].isnull(), 'Closed - Power', 'Closed') # avg. result at system name is the column that has the inference time and power consumption
    return merged_df_edge
# call the merge_dataframes function
merged_df2 = merge_dataframes_edge(df3, df4) # edge

# see what columns are common between merged_df1 and merged_df2, and use those columns to merge the two dataframes
common_columns = list(set(merged_df1.columns) & set(merged_df2.columns))
# print the common columns
print("common columns: ", common_columns)
# merge the two dataframes, with new column called Cloud or Edge, if the system name is in merged_df1, then it is cloud, else it is edge
merged_df1['Cloud or Edge'] = 'Cloud' # add a new column called Cloud or Edge that will say which in the merged are cloud or edge
merged_df2['Cloud or Edge'] = 'Edge'
# merge the two dataframes
merged_df = pd.concat([merged_df1, merged_df2], ignore_index=True)
# save the merged dataframe to a csv file
merged_df.to_csv('merged_inference.csv', index=False)

# print the merged dataframe
print("\n\n")
print("-----------------------------------------------------------------------")
print("merged_df: ")
print(merged_df.head())
print(merged_df.info())

# group by organization and accelerator, where if given organization and accelerator, return the model (system name)
def get_model(organization, accelerator):
    # filter the dataframe by organization and accelerator
    filtered_df = merged_df[(merged_df['Organization'] == organization) & (merged_df['Accelerator'] == accelerator)]
    # return the model
    return filtered_df
# print the get_model function for a specific organization and accelerator: "UntetherAI" and "UntetherAI speedAI240 Slim"
print("\n\n")
print("-----------------------------------------------------------------------")
print("get_model function for UntetherAI and UntetherAI speedAI240 Slim:")
print(get_model('UntetherAI', 'UntetherAI speedAI240 Slim').head())
print("\n\n")
print("-----------------------------------------------------------------------")
# plot power consumption and latency with power consumption on the y axis and latency on the x axis
# get the power consumption and latency columns from the merged dataframe
power_consumption = merged_df['Avg. Result at System Name_x']
latency = merged_df['Avg. Result at System Name_y']
# plot the power consumption and latency
plt.scatter(latency, power_consumption)
plt.xlabel('Latency (s)')
plt.ylabel('Power Consumption (W)')
plt.title('Power Consumption vs Latency')
plt.savefig('power_consumption_vs_latency.png')
plt.show()

# plot the power consumption and latency for each organization and accelerator
def plot_data(df):
    # create a bar plot
    fig, ax = plt.subplots(figsize=(10, 6)) # set the figure size to 10x6 inches
    df.plot(kind='bar', x='System Name (click + for details)', y=['Avg. Result at System Name_x', 'Avg. Result at System Name_y'], ax=ax) 
    plt.title('Inference Time and Power Consumption by Organization and Accelerator')
    plt.xlabel('Organization')
    plt.ylabel('Inference Time (s) / Power Consumption (W)')
    plt.legend(['Inference Time (s)', 'Power Consumption (W)'])
    # Save the plot as a PNG file
    plt.savefig('inference_time_power_consumption.png')
    # plt.show()

# call the plot_data function
plot_data(merged_df)

# the difference between closed and closed - power is that closed - power has the power consumption and latency for each organization and accelerator, while closed only has the latency
# functin