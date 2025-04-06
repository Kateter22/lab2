import pandas as pd
import joblib

test_data = pd.read_csv("test/preprocessed_test_data.csv")
model = joblib.load('model.pkl')

test_data['date'] = pd.to_datetime(test_data['date'])
X_test = test_data[['date']]
X_test['date'] = X_test['date'].astype('int64') // 10**9

predictions = model.predict(X_test)

results = pd.DataFrame({
    'date': test_data['date'],
    'actual_uv_index': test_data['uv_index'],
    'predicted_uv_index': predictions
})

results.to_csv("test/test_results.csv", index=False)

print("Тестирование модели завершено. Результаты сохранены.")
