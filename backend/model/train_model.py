import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def train_model():
    # Load Kaggle CVD Dataset
    print("Loading dataset...")
    try:
        # Path from backend/model to backend/data
        df = pd.read_csv(r'..\data\cardio_train.csv', sep=';')
    except FileNotFoundError:
        print("Error: data/cardio_train.csv not found!")
        print("Please download the Kaggle Cardiovascular Disease dataset")
        print("from: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset")
        return
    
    print(f"Dataset loaded: {df.shape[0]} records")
    
    # Feature engineering
    df['age_years'] = (df['age'] / 365.25).round().astype(int)
    df['bmi'] = df['weight'] / ((df['height']/100) ** 2)
    
    # Features and target
    features = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
                'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    X = df[features]
    y = df['cardio']
    
    print(f"Features: {features}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # Scale numerical features
    numerical_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # Train XGBoost
    print("\nTraining XGBoost...")
    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"{'='*50}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No CVD', 'CVD']))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"\nCross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Save model and scaler in current folder (model/)
    joblib.dump(model, 'xgb_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("\nModel saved to xgb_model.pkl")
    print("Scaler saved to scaler.pkl")
    
    # Save background sample for LIME/DiCE
    background_sample = X_train.sample(n=1000, random_state=42)
    background_sample.to_csv(r'..\data\background_sample.csv', index=False)
    print("Background sample saved to ../data/background_sample.csv")
    
    print("\nTraining complete!")
    return model, scaler

if __name__ == '__main__':
    train_model()