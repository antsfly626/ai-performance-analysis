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
df1 = pd.read_csv('inference_datacenter_closed.csv')
df2 = pd.read_csv('inference_datacenter_closed_power.csv')

# print heads and infos of the dataframes
print("\n\n")
print("-----------------------------------------------------------------------")
print("df1: ")
# print(df1.head())
print(df1.info())
print("\n\n")
print("-----------------------------------------------------------------------")
print("df2: ")
# print(df2.head())
print(df2.info())

# the top row header for each are the same, Public ID,Organization,Availability,System Name (click + for details),# of Nodes,Processor,Accelerator,# of Accelerators,Benchmark,Model MLC,Scenario,Units,Valid / Invalid,# of Accelerators,# of Nodes,Accelerator,Availability,Host Processor Core Count,Processor,Avg. Result at System Name, using this info, I will merge the two dataframes
# i need another column called Division/Power that will say which in the merged are closed or closed - power
# merge the two dataframes
def merge_dataframes(df1, df2):
    # merge the two dataframes
    merged_df = pd.merge(df1, df2, on=['Public ID', 'Organization', 'Availability', 'System Name (click + for details)', '# of Nodes', 'Processor', 'Accelerator', '# of Accelerators', 'Benchmark', 'Model MLC', 'Scenario', 'Units', 'Valid / Invalid', '# of Accelerators', '# of Nodes', 'Accelerator', 'Availability', 'Host Processor Core Count', 'Processor'], how='outer')

    # add a new column called Division/Power that will say which in the merged are closed or closed - power
    merged_df['Division/Power'] = np.where(merged_df['Avg. Result at System Name_x'].isnull(), 'Closed - Power', 'Closed')


    return merged_df
    
# call the merge_dataframes function
merged_df = merge_dataframes(df1, df2)
# save the merged dataframe to a csv file
merged_df.to_csv('merged_inference_datacenter.csv', index=False)

# print the merged dataframe
print("\n\n")
print("-----------------------------------------------------------------------")
print("merged_df: ")
# print(merged_df.head())
print(merged_df.info())

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


print("\n\n")
print("-----------------------------------------------------------------------")

# from merged_df, get new filtered df with only the columns: Organization, Accelerator, 