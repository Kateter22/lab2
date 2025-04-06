import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

train_data = pd.read_csv("train/preprocessed_train_data.csv")

train_data['date'] = pd.to_datetime(train_data['date'])
X = train_data[['date']]
y = train_data['uv_index']

X.loc[:, 'date'] = X['date'].astype(np.int64) // 10**9

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, 'model.pkl')

print("Модель успешно обучена и сохранена.")
