import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import math
import random
from torch.utils.data import Dataset


# Define custom dataset for PyTorch
class EnergyDataset(Dataset):
    def __init__(self, df):
        self.df = df
        # Convert columns to tensors
        self.x_data = torch.tensor(df.index.values, dtype=torch.float32).unsqueeze(1)  # Dummy feature
        self.y_data = torch.tensor(df['y'].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index]
        return x, y

def load_dataset():
    # Load the dataset and filter data for each client
    df = pd.read_csv('dataset/filtered_dataset.csv')
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

    n_clients = len(clients)

    # Randomly select 80% of the clients for training
    train_clients = random.sample(list(clients), math.ceil(0.8*n_clients))

    # Split the data
    train_data = df_reduced[df_reduced['ID Client'].isin(train_clients)]
    test_data = df_reduced[df_reduced['ID Client'].isin(train_clients)]

    print(train_data)
    print(test_data)
    print(train_data.shape, test_data.shape)

    # Return PyTorch datasets
    return EnergyDataset(train_data), EnergyDataset(test_data)
