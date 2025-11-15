# -*- coding: utf-8 -*-
"""
Data loading and preprocessing utilities
"""

import torch
import torch.utils.data
import pandas as pd
import numpy as np

# Ensure device is defined at the top level
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. The MinMaxNorm01 class (This is correct)
class MinMaxNorm01:
    """Scale data to range [0, 1]"""
    
    def __init__(self):
        pass
    
    def fit(self, x):
        # Find min and max along the feature axis (axis=0)
        self.min = x.min(axis=0)
        self.max = x.max(axis=0)
    
    def transform(self, x):
        x = 1.0 * (x - self.min) / (self.max - self.min + 1e-8) # Add epsilon for stability
        return x
    
    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
    
    def inverse_transform(self, x):
        # When inverse transforming, we only care about the target column
        # which we've ensured is at index 0
        x_out = x * (self.max[0] - self.min[0] + 1e-8) + self.min[0]
        return x_out

# 2. The data_loader function (This is correct)
def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    """Create PyTorch DataLoader from tensors"""
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(
        data, 
        batch_size=batch_size,
        shuffle=shuffle, 
        drop_last=drop_last
    )
    return dataloader

# 3. NEW function: load_raw_data
def load_raw_data(csv_file, target_col_name='close'):
    """
    Loads the raw CSV, ensures the target column is at index 0,
    and returns a clean DataFrame.
    """
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    df.index.name = "Date"
    
    if target_col_name not in df.columns:
        raise KeyError(f"Fatal Error: Target column '{target_col_name}' not found.")
        
    try:
        price_index = df.columns.get_loc(target_col_name)
    except KeyError:
        raise KeyError(f"Target column '{target_col_name}' not found in DataFrame.")
        
    print(f"Target column '{target_col_name}' found at index: {price_index}")

    if price_index != 0:
        print(f"Moving target column '{target_col_name}' to index 0 for correct scaling.")
        cols = [target_col_name] + [col for col in df.columns if col != target_col_name]
        df = df[cols]
        price_index = 0

    df.dropna(inplace=True)
    
    num_features = len(df.columns)
    print(f"Data loaded with {num_features} features.")
    
    return df, price_index

# 4. NEW function: create_sequences
def create_sequences(scaled_data, window, predict, price_index):
    """
    Creates time-series sequences from a pre-scaled numpy array.
    """
    X_seq = []
    Y_seq = []
    
    data_len = len(scaled_data)
    num_samples = data_len - window - predict + 1

    for i in range(num_samples):
        x_i = scaled_data[i : i + window, :]
        # y_i is the sequence from the target column (price_index) only
        y_i = scaled_data[i + window : i + window + predict, price_index]
        
        X_seq.append(x_i)
        Y_seq.append(y_i)

    XX = torch.from_numpy(np.array(X_seq)).float().to(device)
    YY = torch.from_numpy(np.array(Y_seq)).float().to(device)
    
    if YY.dim() == 2:
        YY = YY.unsqueeze(-1)

    return XX, YY


# (The old prepare_data function is now replaced by the two new functions above)