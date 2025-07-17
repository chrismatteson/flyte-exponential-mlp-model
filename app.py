import gradio as gr
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Optional

# Global variables to store model and data
model_data = None
training_data = None
metadata = None

def load_model_and_data():
    """Load the trained model and data"""
    global model_data, training_data, metadata
    
    try:
        # Load model (now includes scalers and settings)
        model_path = "./tmp/mlp_model.joblib"
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            print(f"✅ Model data loaded from {model_path}")
            print(f"   Model type: {type(model_data)}")
            if isinstance(model_data, dict):
                print(f"   Contains: {list(model_data.keys())}")
        else:
            print(f"⚠️ Model file not found at {model_path}")
            return False
        
        # Load training data
        data_path = "./tmp/generated_data.csv"
        if os.path.exists(data_path):
            training_data = pd.read_csv(data_path)
            print(f"✅ Training data loaded from {data_path}")
        else:
            print(f"⚠️ Training data not found at {data_path}")
            return False
        
        # Try to load metadata from metrics file
        metrics_path = "./tmp/training_metrics.json"
        if os.path.exists(metrics_path):
            import json
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                metadata = {
                    'formula': metrics.get('formula', 'y = ax² + bx + c'),
                    'x_min': metrics.get('x_min', training_data['x'].min()),
                    'x_max': metrics.get('x_max', training_data['x'].max()),
                    'num_points': len(training_data),
                    'train_r2': metrics.get('train_r2', 0.0),
                    'test_r2': metrics.get('test_r2', 0.0),
                    'feature_engineering': metrics.get('feature_engineering', False),
                    'input_features': metrics.get('input_features', 1),
                    'hidden_layers': metrics.get('hidden_layer_sizes', []),
                    'a': metrics.get('a', 1.0),
                    'b': metrics.get('b', 2.0),
                    'c': metrics.get('c', 1.0),
                    'error': metrics.get('error', 0.0)
                }
            print(f"✅ Metadata loaded from {metrics_path}")
        else:
            # Create basic metadata from data
            metadata = {
                'formula': 'y = ax² + bx + c',
                'x_min': training_data['x'].min(),
                'x_max': training_data['x'].max(),
                'num_points': len(training_data),
                'train_r2': 0.0,
                'test_r2': 0.0,
                'feature_engineering': False,
                'input_features': 1,
                'hidden_layers': []
            }
            print("⚠️ Using basic metadata")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model and data: {str(e)}")
        return False

def preprocess_input(x_value):
    """Apply the same preprocessing as training"""
    global model_data
    
    if model_data is None:
        return None, "Model not loaded"
    
    try:
        # Handle both old format (just model) and new format (dict with model + scalers)
        if isinstance(model_data, dict):
            model = model_data['model']
            scaler_X = model_data.get('scaler_X')
            scaler_y = model_data.get('scaler_y')
            use_feature_engineering = model_data.get('use_feature_engineering', False)
        else:
            # Old format - just the model
            model = model_data
            scaler_X = None
            scaler_y = None
            use_feature_engineering = False
        
        # Prepare input
        x = float(x_value)
        X = np.array([[x]])
        
        # Apply feature engineering if used during training
        if use_feature_engineering:
            X_poly = np.column_stack([
                X.flatten(),           # x
                X.flatten() ** 2,      # x^2
                X.flatten() ** 3,      # x^3
                np.sin(X.flatten()),   # sin(x)
                np.cos(X.flatten()),   # cos(x)
            ])
            X = X_poly
        
        # Apply scaling if used during training
        if scaler_X is not None:
            X_scaled = scaler_X.transform(X)
        else:
            X_scaled = X
        
        # Make prediction
        y_pred_scaled = model.predict(X_scaled)
        
        # Inverse transform if scaling was used
        if scaler_y is not None:
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()[0]
        else:
            y_pred = y_pred_scaled[0]
        
        return y_pred, None
        
    except Exception as e:
        return None, f"Error in preprocessing: {str(e)}"

def predict_y(x_value):
    """Predict y value for given x"""
    global model_data
    
    if model_data is None:
        return "Error: Model not loaded. Please ensure the model file is available."
    
    try:
        prediction, error = preprocess_input(x_value)
        if error:
            return f"Error: {error}"
        return f"Predicted Y: {prediction:.4f}"
    except Exception as e:
        return f"Error: {str(e)}"

def show_data_info():
    """Display information about the training data"""
    global training_data, metadata
    
    if training_data is None or metadata is None:
        return "Error: Data not loaded. Please ensure the data files are available."
    
    try:
        info = f"""
Dataset Information:
- Formula: {metadata['formula']}
- Coefficients: a={metadata.get('a', 'Unknown')}, b={metadata.get('b', 'Unknown')}, c={metadata.get('c', 'Unknown')}
- X Range: {metadata['x_min']} to {metadata['x_max']}
- Number of Points: {metadata['num_points']}
- Error Added: {metadata.get('error', 0.0)}%
- Train R² Score: {metadata['train_r2']:.4f}
- Test R² Score: {metadata['test_r2']:.4f}

Model Architecture:
- Feature Engineering: {metadata['feature_engineering']}
- Input Features: {metadata['input_features']}
- Hidden Layers: {metadata['hidden_layers']}

Data Statistics:
- X Mean: {training_data['x'].mean():.2f}
- X Std: {training_data['x'].std():.2f}
- Y Mean: {training_data['y'].mean():.2f}
- Y Std: {training_data['y'].std():.2f}
- Y Min: {training_data['y'].min():.2f}
- Y Max: {training_data['y'].max():.2f}
        """
        return info
    except Exception as e:
        return f"Error: {str(e)}"

def plot_data_and_prediction(x_value):
    """Plot the data points and prediction"""
    global model_data, training_data, metadata
    
    if model_data is None or training_data is None:
        return None
    
    try:
        x = float(x_value)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot original data
        ax.scatter(training_data['x'], training_data['y'], alpha=0.6, label='Training Data', color='blue', s=30)
        
        # Plot prediction for input point
        prediction, error = preprocess_input(x)
        if error:
            print(f"Error in prediction: {error}")
            return None
            
        ax.scatter(x, prediction, color='red', s=150, label=f'Prediction: ({x}, {prediction:.2f})', zorder=5)
        
        # Plot model predictions over the full range
        x_range = np.linspace(training_data['x'].min(), training_data['x'].max(), 200)
        y_pred_range = []
        
        for x_val in x_range:
            pred, _ = preprocess_input(x_val)
            if pred is not None:
                y_pred_range.append(pred)
            else:
                y_pred_range.append(np.nan)
        
        y_pred_range = np.array(y_pred_range)
        
        # Remove any NaN values
        valid_mask = ~np.isnan(y_pred_range)
        ax.plot(x_range[valid_mask], y_pred_range[valid_mask], 'g-', alpha=0.8, 
                linewidth=2, label='Model Predictions')
        
        # Plot true function using actual coefficients
        if metadata.get('formula') and all(k in metadata for k in ['a', 'b', 'c']):
            # Use the actual coefficients from the training data
            a = metadata['a']
            b = metadata['b'] 
            c = metadata['c']
            y_true = a * x_range**2 + b * x_range + c
            ax.plot(x_range, y_true, 'orange', linestyle='--', alpha=0.7, 
                   linewidth=2, label=f'True Function (y={a}x²+{b}x+{c})')
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_title(f'Advanced MLP Model Predictions\nR² Score: {metadata["test_r2"]:.4f}', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        return fig
    except Exception as e:
        print(f"Error in plotting: {str(e)}")
        return None

def batch_predict(x_values_str):
    """Predict multiple y values for given x values"""
    global model_data
    
    if model_data is None:
        return "Error: Model not loaded."
    
    try:
        # Parse input string (comma-separated values)
        x_values = [float(x.strip()) for x in x_values_str.split(',')]
        
        result = "Batch Predictions:\n"
        for x in x_values:
            prediction, error = preprocess_input(x)
            if error:
                result += f"X={x}: Error - {error}\n"
            else:
                result += f"X={x}: Y={prediction:.4f}\n"
        
        return result
    except Exception as e:
        return f"Error: {str(e)}"

# Load model and data on startup
print("🚀 Starting Advanced MLP Model Predictor...")
if load_model_and_data():
    print("✅ Model and data loaded successfully")
else:
    print("⚠️ Model or data not available - some features may not work")

# Create Gradio interface
with gr.Blocks(title="Advanced MLP Model Predictor") as demo:
    gr.Markdown("# 🧠 Advanced MLP Model Predictor")
    gr.Markdown("This app uses an advanced MLP model with feature engineering to predict Y values for given X inputs.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Single Prediction")
            x_input = gr.Number(label="Enter X value", value=0.0)
            predict_btn = gr.Button("Predict Y", variant="primary")
            result = gr.Textbox(label="Prediction Result")
        
        with gr.Column():
            gr.Markdown("## Batch Prediction")
            batch_input = gr.Textbox(
                label="Enter X values (comma-separated)", 
                value="0, 1, 2, 3",
                placeholder="e.g., 0, 1, 2, 3"
            )
            batch_btn = gr.Button("Batch Predict", variant="secondary")
            batch_result = gr.Textbox(label="Batch Predictions", lines=6)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Model Information")
            info_btn = gr.Button("Show Model Info")
            info_output = gr.Textbox(label="Model and Dataset Information", lines=15)
        
        with gr.Column():
            gr.Markdown("## Advanced Visualization")
            plot_btn = gr.Button("Plot Data and Prediction")
            plot_output = gr.Plot(label="Data and Prediction Plot")
    
    # Event handlers
    predict_btn.click(predict_y, inputs=[x_input], outputs=[result])
    batch_btn.click(batch_predict, inputs=[batch_input], outputs=[batch_result])
    info_btn.click(show_data_info, outputs=[info_output])
    plot_btn.click(plot_data_and_prediction, inputs=[x_input], outputs=[plot_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860) 