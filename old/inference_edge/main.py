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
df1 = pd.read_csv('inference_edge_closed.csv')
df2 = pd.read_csv('inference_edge_closed_power.csv')

# print heads and infos of the dataframes
print("\n\n")
print("-----------------------------------------------------------------------")
print("df1: ")
print(df1.head())
print(df1.info())

print("\n\n")
print("-----------------------------------------------------------------------")
print("df2: ")
print(df2.head())
print(df2.info())
print("\n\n")


# the top row header for each are the same, System Name (click + for details),Accelerator,# of Accelerators,Benchmark1,Model MLC1,Scenario1,Units1,# of Accelerators,# of Nodes,Accelerator,Host Processor Core Count,Processor,Avg. Result at System Name, using this info, I will merge the two dataframes
# i need another column called Division/Power that will say which in the merged are closed or closed - power

# merge the two dataframes
def merge_dataframes(df1, df2):
    # merge the two dataframes
    merged_df = pd.merge(df1, df2, on=['System Name (click + for details)', 'Accelerator', '# of Accelerators', 'Benchmark1', 'Model MLC1', 'Scenario1', 'Units1', '# of Accelerators', '# of Nodes', 'Accelerator', 'Host Processor Core Count', 'Processor'], how='outer')

    # add a new column called Division/Power that will say which in the merged are closed or closed - power
    merged_df['Division/Power'] = np.where(merged_df['Avg. Result at System Name_x'].isnull(), 'Closed - Power', 'Closed')

    # rename the columns to be more descriptive
    # merged_df.rename(columns={'Avg. Result at System Name_x': 'Inference Time (s)', 'Avg. Result at System Name_y': 'Power Consumption (W)'}, inplace=True)

    return merged_df

# call the merge_dataframes function
merged_df = merge_dataframes(df1, df2)


# print the merged dataframe

print("\n\n")
print("-----------------------------------------------------------------------")
print("Merged DataFrame:")
print(merged_df.head())
print(merged_df.info())

# put merged in csv file
merged_df.to_csv('merged.csv', index=False)

# group by organization and accelerator, where if given organization and accelerator, return the model (system name)
def get_model(organization, accelerator):
    # filter the dataframe by organization and accelerator
    filtered_df = merged_df[(merged_df['Organization1'] == organization) & (merged_df['Accelerator'] == accelerator)]
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

def plot_data(df):
    # create a bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(kind='bar', x='System Name (click + for details)', y=['Inference Time (s)', 'Power Consumption (W)'], ax=ax)
    plt.title('Inference Time and Power Consumption by Organization and Accelerator')
    plt.xlabel('Organization')
    plt.ylabel('Inference Time (s) / Power Consumption (W)')
    plt.legend(['Inference Time (s)', 'Power Consumption (W)'])
    # Save the plot as a PNG file
    plt.savefig('inference_time_power_consumption.png')
    # plt.show()

# call the plot_data function
plot_data(merged_df)

print("\n\n")
print("-----------------------------------------------------------------------")

# plot the data for a specific organization and accelerator
def plot_data_for_organization_and_accelerator(df, organization, accelerator):
    # filter the dataframe by organization and accelerator
    filtered_df = df[(df['Organization1'] == organization) & (df['Accelerator'] == accelerator)]
    # create a bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    filtered_df.plot(kind='bar', x='System Name (click + for details)', y=['Inference Time (s)', 'Power Consumption (W)'], ax=ax)
    plt.title(f'Inference Time and Power Consumption for {organization} and {accelerator}')
    plt.xlabel('Model')
    plt.ylabel('Inference Time (s) / Power Consumption (W)')
    plt.legend(['Inference Time (s)', 'Power Consumption (W)'])
    # Save the plot as a PNG file
    plt.savefig(f'{organization}_{accelerator}_inference_time_power_consumption.png')
    # plt.show()

# call the plot_data_for_organization_and_accelerator function
plot_data_for_organization_and_accelerator(merged_df, 'UntetherAI', 'UntetherAI speedAI240 Slim')
print("\n\n")