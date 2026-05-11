from flask import Flask, request, jsonify, render_template_string, send_file, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import shap
import json
from lime.lime_tabular import LimeTabularExplainer
import dice_ml
from datetime import datetime
import tempfile
import os
import warnings
import traceback
warnings.filterwarnings('ignore')

# Try importing PDF libraries with fallbacks
try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False
    print("WeasyPrint not available, will use ReportLab fallback")

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("ReportLab not available")

app = Flask(__name__)
CORS(app)

# ==================== FRONTEND ROUTES ====================

# Serve the main frontend page
@app.route('/')
def home():
    frontend_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend')
    return send_from_directory(frontend_dir, 'index.html')

# Serve CSS, JavaScript, and other frontend assets
@app.route('/<path:filename>')
def frontend_files(filename):
    frontend_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend')
    return send_from_directory(frontend_dir, filename)

# Configuration
API_URL = 'http://localhost:5000'

# Load model and artifacts
print("=" * 60)
print("Loading CardioXAI Model...")
print("=" * 60)

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "model")

    model = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    print("✓ Model loaded successfully")
    print(f"  Model type: {type(model).__name__}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    print("  Please run: python model/train_model.py")
    model = None
    scaler = None

feature_names = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
                 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

# Initialize explainers
explainer = None
lime_explainer = None
dice_exp = None
background_data = None
background_array = None

if model is not None:
    try:
        explainer = shap.TreeExplainer(model)
        print("✓ SHAP explainer initialized")
    except Exception as e:
        print(f"✗ SHAP initialization failed: {e}")

    # Load background data - FIXED
    try:
        background_path = None
        possible_paths = [
            r'data/background_sample.csv',
            'data/background_sample.csv',
            '../data/background_sample.csv',
            './data/background_sample.csv',
            os.path.join(os.path.dirname(__file__), 'data', 'background_sample.csv')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                background_path = path
                break
        
        if background_path:
            background_data = pd.read_csv(background_path)
            print(f"✓ Background data loaded: {background_data.shape}")
            
            # CRITICAL FIX: Ensure only feature columns are used, in correct order
            # Check if 'cardio' target column exists and remove it
            if 'cardio' in background_data.columns:
                background_data = background_data.drop('cardio', axis=1)
                print(f"  Removed target column, new shape: {background_data.shape}")
            
            # Ensure correct column order matches feature_names
            missing_cols = [f for f in feature_names if f not in background_data.columns]
            extra_cols = [c for c in background_data.columns if c not in feature_names]
            
            if missing_cols:
                print(f"  ✗ Missing columns: {missing_cols}")
                raise ValueError(f"Background data missing columns: {missing_cols}")
            
            if extra_cols:
                print(f"  ⚠ Removing extra columns: {extra_cols}")
                background_data = background_data[feature_names]
            else:
                # Reorder to match feature_names exactly
                background_data = background_data[feature_names]
            
            background_array = background_data.values
            print(f"  ✓ Background array shape: {background_array.shape}")
            print(f"  Columns: {list(background_data.columns)}")
            
            # LIME - FIXED with correct data
            try:
                lime_explainer = LimeTabularExplainer(
                    training_data=background_array,
                    feature_names=feature_names,
                    class_names=['No CVD', 'CVD'],
                    mode='classification',
                    discretize_continuous=True,
                    discretizer='quartile',
                    sample_around_instance=True,
                    random_state=42
                )
                print("✓ LIME explainer initialized")
            except Exception as e:
                print(f"✗ LIME initialization failed: {e}")
                import traceback
                traceback.print_exc()
                lime_explainer = None
            
            # DiCE
            try:
                # DiCE needs the target column, so use original data or add predictions
                dice_df = background_data.copy()
                # Generate predictions for DiCE
                dice_scaled = dice_df.copy()
                numerical_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
                dice_scaled[numerical_cols] = scaler.transform(dice_df[numerical_cols])
                dice_df['cardio'] = model.predict(dice_scaled)
                
                dice_data = dice_ml.Data(
                    dataframe=dice_df,
                    continuous_features=['age', 'height', 'weight', 'ap_hi', 'ap_lo'],
                    outcome_name='cardio'
                )
                dice_model = dice_ml.Model(model=model, backend='sklearn')
                dice_exp = dice_ml.Dice(dice_data, dice_model, method='random')
                print("✓ DiCE explainer initialized")
            except Exception as e:
                print(f"✗ DiCE initialization failed: {e}")
                dice_exp = None
        else:
            print("✗ Background data not found at any expected location")
            print("  Expected: data/background_sample.csv")
            print("  Generating synthetic background data...")
            
            # Generate synthetic background data
            np.random.seed(42)
            n_samples = 1000
            
            background_data = pd.DataFrame({
                'age': np.random.randint(10000, 25000, n_samples),
                'gender': np.random.randint(1, 3, n_samples),
                'height': np.random.randint(150, 200, n_samples),
                'weight': np.random.randint(50, 120, n_samples),
                'ap_hi': np.random.randint(90, 180, n_samples),
                'ap_lo': np.random.randint(60, 120, n_samples),
                'cholesterol': np.random.randint(1, 4, n_samples),
                'gluc': np.random.randint(1, 4, n_samples),
                'smoke': np.random.randint(0, 2, n_samples),
                'alco': np.random.randint(0, 2, n_samples),
                'active': np.random.randint(0, 2, n_samples)
            })
            
            # Ensure correct column order
            background_data = background_data[feature_names]
            background_array = background_data.values
            
            # Save for future use
            os.makedirs('data', exist_ok=True)
            background_data.to_csv('data/background_sample.csv', index=False)
            print(f"  ✓ Synthetic background saved: {background_data.shape}")
            
            # Initialize LIME with synthetic data
            try:
                lime_explainer = LimeTabularExplainer(
                    training_data=background_array,
                    feature_names=feature_names,
                    class_names=['No CVD', 'CVD'],
                    mode='classification',
                    discretize_continuous=True,
                    discretizer='quartile',
                    sample_around_instance=True,
                    random_state=42
                )
                print("✓ LIME explainer initialized with synthetic data")
            except Exception as e:
                print(f"✗ LIME initialization failed: {e}")
                lime_explainer = None
            
    except Exception as e:
        print(f"✗ Error loading background data: {e}")
        import traceback
        traceback.print_exc()
        background_data = None
        background_array = None

print("=" * 60)
print("System Ready!" if model else "System NOT Ready - Train model first")
print("=" * 60)

# ==================== ROUTES ====================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'shap_available': explainer is not None,
        'lime_available': lime_explainer is not None,
        'dice_available': dice_exp is not None,
        'weasyprint_available': WEASYPRINT_AVAILABLE,
        'reportlab_available': REPORTLAB_AVAILABLE,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/debug/explainers', methods=['GET'])
def debug_explainers():
    """Debug endpoint for explainers"""
    return jsonify({
        'model_loaded': model is not None,
        'shap_available': explainer is not None,
        'lime_available': lime_explainer is not None,
        'dice_available': dice_exp is not None,
        'background_data_loaded': background_data is not None,
        'background_array_shape': background_array.shape if background_array is not None else None,
        'feature_names': feature_names,
        'background_columns': list(background_data.columns) if background_data is not None else None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint with SHAP and LIME explanations"""
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please train the model first.'
        }), 503
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = feature_names
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({
                'success': False,
                'error': f'Missing fields: {missing}'
            }), 400
        
        # Preprocess input
        input_df, input_scaled = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # SHAP Explanation
        shap_result = compute_shap(input_scaled, input_df)
        
        # LIME Explanation - FIXED
        lime_result = compute_lime(input_scaled)
        
        # Calculate metrics
        bmi = input_df['weight'].iloc[0] / ((input_df['height'].iloc[0]/100) ** 2)
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'probability': {
                'no_cvd': float(probability[0]),
                'cvd': float(probability[1])
            },
            'risk_level': get_risk_level(probability[1]),
            'shap_explanation': shap_result,
            'lime_explanation': lime_result.get('explanation', []) if not lime_result.get('error') else [],
            'lime_error': lime_result.get('error'),
            'lime_available': lime_result.get('error') is None,
            'patient_metrics': {
                'bmi': round(bmi, 1),
                'bp_category': get_bp_category(input_df['ap_hi'].iloc[0], input_df['ap_lo'].iloc[0]),
                'age_years': int(input_df['age'].iloc[0] / 365.25)
            },
            'input_data': data
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/counterfactual', methods=['POST'])
def generate_counterfactuals():
    """Generate counterfactual explanations using DiCE"""
    if dice_exp is None:
        return jsonify({
            'success': False,
            'error': 'Counterfactual generation not available. Check if model is trained.'
        }), 503
    
    try:
        data = request.get_json()
        input_df, input_scaled = preprocess_input(data)
        
        # Generate counterfactuals
        query_instance = input_scaled
        
        cf_explanations = dice_exp.generate_counterfactuals(
            query_instance,
            total_CFs=3,
            desired_class=0,  # Target: No CVD
            features_to_vary=['cholesterol', 'ap_hi', 'ap_lo', 'weight', 'smoke', 'alco', 'active'],
            permitted_range={
                'cholesterol': [1, 3],
                'ap_hi': [90, 160],
                'ap_lo': [60, 100],
                'weight': [40, 150],
                'smoke': [0, 1],
                'alco': [0, 1],
                'active': [0, 1]
            }
        )
        
        counterfactuals = []
        original_prob = float(model.predict_proba(input_scaled)[0][1])
        
        for i, cf in enumerate(cf_explanations.cf_examples_list[0].final_cfs_df.itertuples()):
            cf_dict = {}
            changes = []
            
            for feat in feature_names:
                orig_val = float(input_df.iloc[0][feat])
                cf_val = float(getattr(cf, feat))
                
                if abs(orig_val - cf_val) > 0.01:
                    change_desc = format_change(feat, orig_val, cf_val)
                    changes.append({
                        'feature': feat,
                        'original': orig_val,
                        'new': cf_val,
                        'description': change_desc
                    })
                    cf_dict[feat] = cf_val
            
            # Calculate new probability
            if changes:
                cf_df = pd.DataFrame([cf_dict])
                cf_scaled = cf_df.copy()
                numerical_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
                cf_scaled[numerical_cols] = scaler.transform(cf_df[numerical_cols])
                new_prob = float(model.predict_proba(cf_scaled)[0][1])
            else:
                new_prob = original_prob
            
            counterfactuals.append({
                'id': i + 1,
                'changes': changes,
                'original_risk': original_prob,
                'new_risk': new_prob,
                'risk_reduction': original_prob - new_prob,
                'success': new_prob < 0.5
            })
        
        # Sort by risk reduction
        counterfactuals.sort(key=lambda x: x['risk_reduction'], reverse=True)
        
        return jsonify({
            'success': True,
            'original_risk': original_prob,
            'counterfactuals': counterfactuals,
            'summary': generate_cf_summary(counterfactuals[0]) if counterfactuals else None
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/whatif', methods=['POST'])
def what_if_analysis():
    """Simple what-if scenario analysis"""
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 503
    
    try:
        data = request.get_json()
        base_data = data.get('base_data', {})
        modifications = data.get('modifications', {})
        
        # Apply modifications
        modified_data = base_data.copy()
        modified_data.update(modifications)
        
        # Predict
        input_df, input_scaled = preprocess_input(modified_data)
        new_probability = model.predict_proba(input_scaled)[0][1]
        
        return jsonify({
            'success': True,
            'modified_probability': float(new_probability),
            'modifications': modifications,
            'original_probability': data.get('original_probability', 0),
            'risk_change': float(new_probability - data.get('original_probability', 0))
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/generate-report', methods=['POST'])
def generate_pdf_report():
    """Generate PDF report using available library"""
    try:
        data = request.get_json()
        
        # Check what PDF engine to use
        if WEASYPRINT_AVAILABLE:
            print("Using WeasyPrint for PDF generation")
            return generate_weasyprint_pdf(data)
        elif REPORTLAB_AVAILABLE:
            print("Using ReportLab for PDF generation")
            return generate_reportlab_pdf(data)
        else:
            return jsonify({
                'success': False,
                'error': 'No PDF generation library available. Install weasyprint or reportlab.'
            }), 503
            
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

# ==================== HELPER FUNCTIONS ====================

def preprocess_input(data):
    """Convert input dict to DataFrame and scale features"""
    input_df = pd.DataFrame([{
        'age': float(data['age']),
        'gender': int(data['gender']),
        'height': float(data['height']),
        'weight': float(data['weight']),
        'ap_hi': float(data['ap_hi']),
        'ap_lo': float(data['ap_lo']),
        'cholesterol': int(data['cholesterol']),
        'gluc': int(data['gluc']),
        'smoke': int(data['smoke']),
        'alco': int(data['alco']),
        'active': int(data['active'])
    }])
    
    numerical_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
    input_scaled = input_df.copy()
    input_scaled[numerical_cols] = scaler.transform(input_df[numerical_cols])
    
    return input_df, input_scaled

def compute_shap(input_scaled, input_df):
    """Compute SHAP values and format for output"""
    if explainer is None:
        return {'error': 'SHAP not available'}
    
    shap_values = explainer.shap_values(input_scaled)
    
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
            'value': float(input_df.iloc[0][feat]),
            'contribution': float(shap_for_class[i]),
            'description': get_feature_description(feat)
        })
    
    contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
    
    return {
        'base_value': float(base_value),
        'contributions': contributions,
        'total_value': float(base_value + sum(shap_for_class))
    }

def compute_lime(input_scaled):
    """Compute LIME explanation - FIXED"""
    global lime_explainer
    
    if lime_explainer is None:
        print("LIME explainer not initialized")
        return {
            'error': 'LIME not available - explainer not initialized',
            'explanation': []
        }
    
    try:
        # Ensure input is proper format
        if hasattr(input_scaled, 'values'):
            instance = input_scaled.values[0]
        else:
            instance = np.array(input_scaled).flatten()
        
        # Validate instance shape matches background
        if len(instance) != len(feature_names):
            print(f"Shape mismatch: instance={len(instance)}, expected={len(feature_names)}")
            return {
                'error': f'Feature count mismatch: got {len(instance)}, expected {len(feature_names)}',
                'explanation': []
            }
        
        print(f"Computing LIME for instance: {instance[:3]}... (shape: {instance.shape})")
        
        # Generate explanation
        lime_exp = lime_explainer.explain_instance(
            data_row=instance,
            predict_fn=model.predict_proba,
            num_features=8,           # Top 8 features
            top_labels=1,             # Explain positive class
            num_samples=5000          # More samples for stability
        )
        
        # Extract explanation for CVD class (class 1)
        label = 1
        explanation_list = lime_exp.as_list(label=label)
        
        # Format for frontend
        lime_explanation = []
        for feature_desc, weight in explanation_list:
            lime_explanation.append({
                'feature_range': feature_desc,
                'feature_name': extract_feature_name(feature_desc),
                'weight': float(weight),
                'direction': 'increases risk' if weight > 0 else 'decreases risk',
                'abs_weight': abs(float(weight))
            })
        
        # Sort by absolute importance
        lime_explanation.sort(key=lambda x: x['abs_weight'], reverse=True)
        
        print(f"✓ LIME generated {len(lime_explanation)} explanations")
        return {
            'error': None,
            'explanation': lime_explanation
        }
        
    except Exception as e:
        print(f"LIME computation error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'error': str(e),
            'explanation': []
        }

def extract_feature_name(feature_desc):
    """Extract clean feature name from LIME description like 'age > 15000.00'"""
    for feat in feature_names:
        if feat in feature_desc:
            return feat
    # Fallback: return first word
    return feature_desc.split()[0] if ' ' in feature_desc else feature_desc
          
def format_change(feat, orig, new):
    """Format feature change for human reading"""
    if feat == 'cholesterol':
        levels = {1: 'Normal', 2: 'Above Normal', 3: 'Well Above Normal'}
        return f"Cholesterol: {levels.get(int(orig), orig)} → {levels.get(int(new), new)}"
    elif feat == 'ap_hi':
        return f"Systolic BP: {int(orig)} → {int(new)} mmHg"
    elif feat == 'ap_lo':
        return f"Diastolic BP: {int(orig)} → {int(new)} mmHg"
    elif feat == 'weight':
        return f"Weight: {orig:.1f} → {new:.1f} kg"
    elif feat == 'smoke':
        return "Stop smoking" if new == 0 else "Start smoking"
    elif feat == 'alco':
        return "Reduce alcohol" if new == 0 else "Increase alcohol"
    elif feat == 'active':
        return "Become physically active" if new == 1 else "Reduce activity"
    return f"{feat}: {orig} → {new}"

def generate_cf_summary(best_cf):
    """Generate human-readable summary of best counterfactual"""
    if not best_cf or not best_cf.get('changes'):
        return None
    
    changes_text = [c['description'] for c in best_cf['changes']]
    new_risk_pct = int(best_cf['new_risk'] * 100)
    
    return {
        'action_items': changes_text,
        'outcome': f"If you make these changes, your risk drops to {new_risk_pct}%",
        'impact': f"Risk reduction of {int(best_cf['risk_reduction'] * 100)} percentage points"
    }

def get_risk_level(probability):
    """Get risk level description"""
    if probability < 0.3:
        return {'level': 'low', 'color': 'green', 'message': 'Low risk - Continue healthy lifestyle'}
    elif probability < 0.7:
        return {'level': 'moderate', 'color': 'orange', 'message': 'Moderate risk - Schedule check-up within 3 months'}
    else:
        return {'level': 'high', 'color': 'red', 'message': 'High risk - Consult cardiologist immediately'}

def get_bp_category(systolic, diastolic):
    """Get blood pressure category"""
    if systolic < 120 and diastolic < 80:
        return 'Normal'
    elif systolic < 130 and diastolic < 80:
        return 'Elevated'
    elif systolic < 140 or diastolic < 90:
        return 'Stage 1 Hypertension'
    else:
        return 'Stage 2 Hypertension'

def get_feature_description(feature):
    """Get human-readable feature description"""
    descriptions = {
        'age': 'Patient age (days)',
        'gender': 'Gender (1=Female, 2=Male)',
        'height': 'Height in cm',
        'weight': 'Weight in kg',
        'ap_hi': 'Systolic blood pressure',
        'ap_lo': 'Diastolic blood pressure',
        'cholesterol': 'Cholesterol level (1=Normal, 2=Above Normal, 3=High)',
        'gluc': 'Glucose level (1=Normal, 2=Above Normal, 3=High)',
        'smoke': 'Smoking status (0=No, 1=Yes)',
        'alco': 'Alcohol consumption (0=No, 1=Yes)',
        'active': 'Physical activity (0=No, 1=Yes)'
    }
    return descriptions.get(feature, feature)

def get_top_features(shap_data, n=3):
    """Extract top N features from SHAP data"""
    contribs = shap_data.get('contributions', [])
    return contribs[:n]

# ==================== PDF GENERATION ====================

def generate_weasyprint_pdf(data):
    """Generate PDF using WeasyPrint (best quality, requires system deps)"""
    from jinja2 import Template
    
    patient_data = data.get('patient_data', {})
    prediction_data = data.get('prediction', {})
    shap_data = data.get('shap_explanation', {})
    cf_data = data.get('counterfactuals', {})
    metrics = data.get('patient_metrics', {})
    
    # HTML Template
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Cardiovascular Risk Assessment Report</title>
        <style>
            @page { size: A4; margin: 20mm; }
            body { font-family: 'Helvetica', 'Arial', sans-serif; margin: 0; padding: 0; color: #333; line-height: 1.6; font-size: 11pt; }
            .header { text-align: center; border-bottom: 3px solid #2563eb; padding-bottom: 20px; margin-bottom: 30px; }
            .logo { font-size: 48px; margin-bottom: 10px; }
            h1 { color: #1e40af; margin: 0; font-size: 24pt; }
            .subtitle { color: #64748b; font-size: 10pt; margin-top: 5px; }
            .risk-section { background: #f8fafc; border-radius: 12px; padding: 30px; margin: 20px 0; text-align: center; }
            .risk-score { font-size: 60pt; font-weight: bold; margin: 20px 0; }
            .risk-high { color: #dc2626; }
            .risk-moderate { color: #ea580c; }
            .risk-low { color: #16a34a; }
            .section { margin: 25px 0; }
            h2 { color: #1e40af; border-left: 4px solid #2563eb; padding-left: 15px; font-size: 14pt; margin-top: 0; }
            .feature-table { width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 10pt; }
            .feature-table th { background: #e0e7ff; padding: 10px; text-align: left; font-weight: 600; }
            .feature-table td { padding: 8px 10px; border-bottom: 1px solid #e2e8f0; }
            .positive { color: #dc2626; font-weight: 600; }
            .negative { color: #16a34a; font-weight: 600; }
            .cf-box { background: #fef3c7; border: 2px solid #f59e0b; border-radius: 8px; padding: 20px; margin: 15px 0; }
            .cf-title { font-weight: bold; color: #92400e; margin-bottom: 10px; font-size: 12pt; }
            .action-item { background: white; padding: 8px 12px; margin: 5px 0; border-radius: 4px; border-left: 3px solid #f59e0b; font-size: 10pt; }
            .metrics-grid { display: flex; justify-content: space-around; margin-top: 20px; flex-wrap: wrap; }
            .metric-box { text-align: center; padding: 10px; }
            .metric-label { font-size: 9pt; color: #64748b; display: block; }
            .metric-value { font-size: 14pt; font-weight: bold; color: #1e293b; }
            .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #e2e8f0; font-size: 9pt; color: #64748b; text-align: center; }
            .disclaimer { background: #fee2e2; border: 1px solid #ef4444; padding: 12px; border-radius: 6px; margin-top: 30px; font-size: 9pt; color: #991b1b; }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="logo">🫀</div>
            <h1>Cardiovascular Risk Assessment Report</h1>
            <div class="subtitle">Generated on {{ generated_date }} | AI-Powered Analysis</div>
        </div>

        <div class="risk-section">
            <h3>Current Risk Assessment</h3>
            {% set prob = prediction.probability.cvd * 100 %}
            <div class="risk-score {% if prob > 70 %}risk-high{% elif prob > 30 %}risk-moderate{% else %}risk-low{% endif %}">
                {{ "%.0f"|format(prob) }}%
            </div>
            <p><strong>{{ prediction.risk_level.level.upper() }} RISK</strong></p>
            <p>{{ prediction.risk_level.message }}</p>
            
            <div class="metrics-grid">
                <div class="metric-box">
                    <span class="metric-label">Age</span>
                    <span class="metric-value">{{ patient_metrics.age_years }} years</span>
                </div>
                <div class="metric-box">
                    <span class="metric-label">BMI</span>
                    <span class="metric-value">{{ "%.1f"|format(patient_metrics.bmi) }}</span>
                </div>
                <div class="metric-box">
                    <span class="metric-label">Blood Pressure</span>
                    <span class="metric-value">{{ patient_metrics.bp_category }}</span>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>🔍 Key Risk Factors</h2>
            <p style="font-size: 10pt; color: #64748b;">Based on SHAP analysis, these features most influenced your risk score:</p>
            <table class="feature-table">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Feature</th>
                        <th>Your Value</th>
                        <th>Impact on Risk</th>
                    </tr>
                </thead>
                <tbody>
                    {% for feat in top_features %}
                    <tr>
                        <td>{{ loop.index }}</td>
                        <td>{{ feat.feature }}</td>
                        <td>{{ feat.value }}</td>
                        <td class="{% if feat.contribution > 0 %}positive{% else %}negative{% endif %}">
                            {% if feat.contribution > 0 %}+{% endif %}{{ "%.3f"|format(feat.contribution) }}
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        {% if cf_summary and cf_summary.outcome %}
        <div class="section">
            <h2>💡 Personalized Recommendations</h2>
            <div class="cf-box">
                <div class="cf-title">How to Reduce Your Risk</div>
                <p style="font-size: 11pt; margin-bottom: 15px;"><strong>{{ cf_summary.outcome }}</strong></p>
                <p style="font-size: 10pt; color: #92400e; margin-bottom: 15px;"><em>{{ cf_summary.impact }}</em></p>
                
                <div>
                    <strong style="font-size: 10pt;">Recommended Actions:</strong>
                    {% for action in cf_summary.action_items %}
                    <div class="action-item">{{ action }}</div>
                    {% endfor %}
                </div>
            </div>
        </div>
        {% endif %}

        <div class="section">
            <h2>📊 Complete Health Profile</h2>
            <table class="feature-table">
                <thead>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                        <th>Reference Range</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td>Age</td><td>{{ patient_metrics.age_years }} years</td><td>Adult</td></tr>
                    <tr><td>Gender</td><td>{{ 'Male' if patient.gender == 2 else 'Female' }}</td><td>-</td></tr>
                    <tr><td>Height</td><td>{{ patient.height }} cm</td><td>-</td></tr>
                    <tr><td>Weight</td><td>{{ patient.weight }} kg</td><td>-</td></tr>
                    <tr><td>BMI</td><td>{{ "%.1f"|format(patient_metrics.bmi) }}</td><td>18.5 - 24.9 (Normal)</td></tr>
                    <tr><td>Systolic BP</td><td>{{ patient.ap_hi }} mmHg</td><td>&lt; 120 mmHg (Normal)</td></tr>
                    <tr><td>Diastolic BP</td><td>{{ patient.ap_lo }} mmHg</td><td>&lt; 80 mmHg (Normal)</td></tr>
                    <tr><td>Cholesterol</td><td>{{ 'Normal' if patient.cholesterol == 1 else 'Above Normal' if patient.cholesterol == 2 else 'High' }}</td><td>Normal</td></tr>
                    <tr><td>Glucose</td><td>{{ 'Normal' if patient.gluc == 1 else 'Above Normal' if patient.gluc == 2 else 'High' }}</td><td>Normal</td></tr>
                    <tr><td>Smoking</td><td>{{ 'Yes' if patient.smoke == 1 else 'No' }}</td><td>No</td></tr>
                    <tr><td>Alcohol</td><td>{{ 'Yes' if patient.alco == 1 else 'No' }}</td><td>No</td></tr>
                    <tr><td>Physical Activity</td><td>{{ 'Yes' if patient.active == 1 else 'No' }}</td><td>Yes</td></tr>
                </tbody>
            </table>
        </div>

        <div class="disclaimer">
            <strong>Medical Disclaimer:</strong> This report is generated by an AI system for educational purposes only. 
            It does not constitute medical advice, diagnosis, or treatment. Always consult with a qualified healthcare 
            provider for medical decisions.
        </div>

        <div class="footer">
            <p><strong>CardioXAI - Explainable AI for Healthcare</strong></p>
            <p>This report uses SHAP and DiCE explainability techniques.</p>
            <p style="margin-top: 10px;">© {{ generated_date.split(' ')[-1] }} CardioXAI. All rights reserved.</p>
        </div>
    </body>
    </html>
    """
    
    template = Template(html_template)
    html_content = template.render(
        patient=patient_data,
        prediction=prediction_data,
        patient_metrics=metrics,
        shap=shap_data,
        top_features=get_top_features(shap_data, 5),
        cf_summary=cf_data.get('summary', {}) if cf_data else {},
        generated_date=datetime.now().strftime("%B %d, %Y")
    )
    
    # Generate PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        pdf_path = tmp.name
    
    HTML(string=html_content).write_pdf(pdf_path)
    
    return send_file(
        pdf_path,
        as_attachment=True,
        download_name=f"CVD_Risk_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
        mimetype='application/pdf'
    )

def generate_reportlab_pdf(data):
    """Generate PDF using ReportLab (pure Python, works everywhere)"""
    patient_data = data.get('patient_data', {})
    prediction_data = data.get('prediction', {})
    shap_data = data.get('shap_explanation', {})
    metrics = data.get('patient_metrics', {})
    cf_data = data.get('counterfactuals', {})
    
    # Create PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        pdf_path = tmp.name
    
    doc = SimpleDocTemplate(
        pdf_path, 
        pagesize=A4,
        rightMargin=72, 
        leftMargin=72,
        topMargin=72, 
        bottomMargin=18
    )
    
    elements = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#1e40af'),
        spaceAfter=12,
        alignment=1
    )
    
    # Header
    elements.append(Paragraph("🫀 Cardiovascular Risk Assessment Report", title_style))
    elements.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y')}", 
                             styles['Normal']))
    elements.append(Spacer(1, 0.5*cm))
    
    # Risk Score Box
    prob = prediction_data.get('probability', {}).get('cvd', 0)
    risk_level = prediction_data.get('risk_level', {})
    risk_color = colors.HexColor('#dc2626') if prob > 0.7 else colors.HexColor('#ea580c') if prob > 0.3 else colors.HexColor('#16a34a')
    
    risk_data = [
        ['RISK ASSESSMENT'],
        [f"{int(prob * 100)}%"],
        [risk_level.get('level', 'Unknown').upper()],
        [risk_level.get('message', '')]
    ]
    
    risk_table = Table(risk_data, colWidths=[16*cm])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e0e7ff')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('FONTSIZE', (0, 1), (-1, 1), 36),
        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0, 1), (-1, 1), risk_color),
        ('FONTSIZE', (0, 2), (-1, 2), 14),
        ('FONTNAME', (0, 2), (-1, 2), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 1), (-1, 1), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    elements.append(risk_table)
    elements.append(Spacer(1, 0.5*cm))
    
    # Metrics
    metrics_data = [
        ['Age', 'BMI', 'BP Category'],
        [f"{metrics.get('age_years', 'N/A')} years", 
         str(metrics.get('bmi', 'N/A')),
         metrics.get('bp_category', 'N/A')]
    ]
    
    metrics_table = Table(metrics_data, colWidths=[5.3*cm, 5.3*cm, 5.3*cm])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
    ]))
    
    elements.append(metrics_table)
    elements.append(Spacer(1, 0.5*cm))
    
    # Key Risk Factors
    elements.append(Paragraph("Key Risk Factors", styles['Heading2']))
    
    contributions = shap_data.get('contributions', [])[:5]
    if contributions:
        feature_data = [['Feature', 'Value', 'Impact']]
        for feat in contributions:
            impact = feat['contribution']
            impact_str = f"+{impact:.3f}" if impact > 0 else f"{impact:.3f}"
            feature_data.append([
                feat['feature'],
                str(feat['value']),
                impact_str
            ])
        
        feature_table = Table(feature_data, colWidths=[8*cm, 4*cm, 4*cm])
        feature_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e0e7ff')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
        ]))
        
        elements.append(feature_table)
    
    elements.append(Spacer(1, 0.5*cm))
    
    # Recommendations
    cf_summary = cf_data.get('summary', {}) if cf_data else {}
    if cf_summary and cf_summary.get('outcome'):
        elements.append(Paragraph("Personalized Recommendations", styles['Heading2']))
        elements.append(Paragraph(cf_summary.get('outcome', ''), styles['Normal']))
        elements.append(Spacer(1, 0.2*cm))
        
        for action in cf_summary.get('action_items', []):
            elements.append(Paragraph(f"• {action}", styles['Normal']))
        
        elements.append(Spacer(1, 0.2*cm))
        elements.append(Paragraph(cf_summary.get('impact', ''), 
                                 ParagraphStyle('Impact', parent=styles['Normal'], 
                                               textColor=colors.HexColor('#92400e'))))
    
    elements.append(Spacer(1, 1*cm))
    
    # Disclaimer
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#991b1b'),
        backColor=colors.HexColor('#fee2e2'),
        borderPadding=10,
        alignment=1
    )
    
    elements.append(Paragraph(
        "Medical Disclaimer: This report is for educational purposes only and does not constitute medical advice. "
        "Always consult with a qualified healthcare provider.", 
        disclaimer_style
    ))
    
    # Build PDF
    doc.build(elements)
    
    return send_file(
        pdf_path,
        as_attachment=True,
        download_name=f"CVD_Risk_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
        mimetype='application/pdf'
    )

# ==================== MAIN ====================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Starting CardioXAI Server...")
    print(f"API URL: http://localhost:5000")
    print(f"Health Check: http://localhost:5000/health")
    print(f"Debug: http://localhost:5000/debug/explainers")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)