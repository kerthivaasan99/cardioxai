import shap
import numpy as np

def get_shap_explainer(model):
    """Initialize SHAP TreeExplainer for XGBoost"""
    return shap.TreeExplainer(model)

def explain_prediction(explainer, input_data, feature_names):
    """Generate SHAP explanation for a single prediction"""
    shap_values = explainer.shap_values(input_data)
    
    # Handle binary classification
    if isinstance(shap_values, list):
        shap_for_class = shap_values[1][0]
        base_value = explainer.expected_value[1]
    else:
        shap_for_class = shap_values[0]
        base_value = explainer.expected_value
    
    contributions = []
    for i, feat in enumerate(feature_names):
        contributions.append({
            'feature': feat,
            'value': float(input_data.iloc[0][i]),
            'contribution': float(shap_for_class[i])
        })
    
    return {
        'base_value': float(base_value),
        'contributions': sorted(contributions, key=lambda x: abs(x['contribution']), reverse=True),
        'total_value': float(base_value + sum(shap_for_class))
    }