import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(file_path):
    data = pd.read_csv(file_path)

    data['date'] = pd.to_datetime(data['date'])

    scaler = StandardScaler()
    data['uv_index'] = scaler.fit_transform(data[['uv_index']])
    
    return data

train_data = preprocess_data("train/train_data.csv")
test_data = preprocess_data("test/test_data.csv")
train_data.to_csv("train/preprocessed_train_data.csv", index=False)
test_data.to_csv("test/preprocessed_test_data.csv", index=False)
