import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def load_and_preprocess_data(filepath):
    """Load and preprocess the cardiovascular dataset"""
    df = pd.read_csv(filepath, sep=';')
    
    features = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
                'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    
    X = df[features]
    y = df['cardio']
    
    return X, y

def scale_features(X, scaler=None, fit=False):
    """Scale numerical features"""
    numerical_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
    
    if fit:
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        return X, scaler
    else:
        X[numerical_cols] = scaler.transform(X[numerical_cols])
        return X