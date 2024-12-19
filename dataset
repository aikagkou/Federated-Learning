import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# Load the dataset and filter data for each client
df = pd.read_csv('/content/daily_dataset.csv')
df = df[['LCLid', 'day', 'energy_median']]
df.columns = ['ID Client', 'Date', 'Energy_Median']
df['Date'] = pd.to_datetime(df['Date'])

# Data reduction steps to filter out 80% of unique dates
unique_dates = df['Date'].unique()
num_dates_to_remove = int(0.80 * len(unique_dates))
dates_to_remove = np.random.choice(unique_dates, num_dates_to_remove, replace=False)
df_reduced = df[~df['Date'].isin(dates_to_remove)]
print(df_reduced)

# Renaming for LSTM input format and fetching unique clients
df_reduced = df_reduced.rename(columns={"Date": "ds", "Energy_Median": "y"})
clients = df_reduced["ID Client"].unique()
print(clients)

scaler = MinMaxScaler(feature_range=(0, 1))
df_reduced['y'] = scaler.fit_transform(df_reduced[['y']])
