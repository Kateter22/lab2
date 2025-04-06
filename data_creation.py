import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

def generate_data(num_samples=100, anomalies=False):
    dates = [datetime.now() - timedelta(days=i) for i in range(num_samples)]
    uv_index = np.random.uniform(0, 15, num_samples)

    if anomalies:
        anomaly_indices = np.random.choice(num_samples, size=5, replace=False)
        for index in anomaly_indices:
            uv_index[index] = np.random.uniform(200, 3000) # Аномальные значения

    data = pd.DataFrame({
        'date': dates,
        'uv_index': uv_index
    })
    
    return data

os.makedirs("train", exist_ok=True)
os.makedirs("test", exist_ok=True)
train_data = generate_data(100, anomalies=True)
test_data = generate_data(50)
train_data.to_csv("train/train_data.csv", index=False)
test_data.to_csv("test/test_data.csv", index=False)
