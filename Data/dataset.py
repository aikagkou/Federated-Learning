import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def load_dataset():
    # Load the dataset and filter data for each client
    df = pd.read_csv('Data/filtered_dataset.csv')
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
    
    # Split the data for each client
    train_data_list = []
    test_data_list = []
    
    for client in clients:
        client_data = df_reduced[df_reduced["ID Client"] == client]
        
        # Sort by date to maintain time order
        client_data = client_data.sort_values(by="ds")
        
        # Perform the train-test split
        train_data, test_data = train_test_split(
            client_data,
            test_size=0.2,
            shuffle=False  # Important for time series data
        )
        
        train_data_list.append(train_data)
        test_data_list.append(test_data)
    
    # Combine all clients' data back together if needed
    train_data_combined = pd.concat(train_data_list)
    test_data_combined = pd.concat(test_data_list)
    
    print("Train Data:")
    print(train_data_combined.head())
    
    print("\nTest Data:")
    print(test_data_combined.head())

    return train_data_combined, test_data_combined
