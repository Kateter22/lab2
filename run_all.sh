    @echo off
    pip install numpy scikit-learn joblib pandas
    python3 data_creation.py
    python3 model_preprocessing.py
    python3 model_training.py
    python3 model_testing.py
    pause
   