from flytekit import workflow, task, Resources, ImageSpec, map_task, current_context
from flytekit.types.file import FlyteFile
from flytekit.types.schema import FlyteSchema
from union import ActorEnvironment
import union
from flytekitplugins.deck.renderer import ImageRenderer, MarkdownRenderer
import pandas as pd
from typing_extensions import Annotated

# Union AI Artifacts for serverless deployment
ModelArtifact = union.Artifact(name="mlp_model_artifact")
DataArtifact = union.Artifact(name="training_data_artifact") 
MetricsArtifact = union.Artifact(name="training_metrics_artifact")
ValidationArtifact = union.Artifact(name="validation_report_artifact")

class ResponsiveImageRenderer(ImageRenderer):
    """Enhanced ImageRenderer that creates responsive images that auto-resize to page width"""
    
    @staticmethod
    def _image_to_html_string(img: "PIL.Image.Image") -> str:
        import base64
        from io import BytesIO

        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Enhanced HTML with responsive CSS styling
        return f"""
        <div style="width: 100%; text-align: center; margin: 20px 0;">
            <img src="data:image/png;base64,{img_base64}" 
                 alt="Rendered Image"
                 style="max-width: 100%; 
                        height: auto; 
                        border: 1px solid #ddd; 
                        border-radius: 8px; 
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                        background: white;
                        display: block;
                        margin: 0 auto;">
        </div>
        """
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from typing import Tuple, List, Dict, Any, Union, Annotated
from dataclasses import dataclass
import json
import os

@dataclass
class DataMetadata:
    x_min: int
    x_max: int  
    error: float
    a: float
    b: float
    c: float
    num_points: int
    formula: str

# Grid search for hyperparameter optimization
# Note: This is a custom grid search implementation, not using Optuna

image_spec = ImageSpec(
    builder="union",
    registry=os.environ.get("IMAGE_SPEC_REGISTRY"),
    packages=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "flytekitplugins-deck-standard",
        "markdown",
        "seaborn>=0.12.0"
    ],
)

actor = ActorEnvironment(
    name="actor",
    ttl_seconds=30,
    container_image=image_spec,
    requests=Resources(cpu="1", mem="2Gi"),
    limits=Resources(cpu="1", mem="2Gi")
)

@actor.task(
    cache=True,  # Cache data generation since it's deterministic
    cache_version="1.5"
)
def generate_data(
    x_min: int = 0,
    x_max: int = 100,
    error: float = 0.0,
    a: float = 1.0,
    b: float = 2.0,
    c: float = 1.0
) -> Tuple[FlyteFile, DataMetadata]:
    """
    Generate synthetic data using the formula y = ax^2 + bx + c
    
    Args:
        x_min: Minimum x value (default: -10)
        x_max: Maximum x value (default: 10)
        error: Error percentage to add to y values (default: 0.0)
        a: Coefficient for x^2 term (default: 1.0)
        b: Coefficient for x term (default: 2.0)
        c: Constant term (default: 1.0)
    
    Returns:
        Tuple of (data file path, metadata dictionary)
    """
    # Generate x values
    x_values = list(range(x_min, x_max + 1))
    
    # Generate y values using the formula y = ax^2 + bx + c
    y_values = []
    for x in x_values:
        y = a * x**2 + b * x + c
        
        # Add error if specified
        if error > 0:
            # Calculate how many values to modify based on error percentage
            num_to_modify = int(len(x_values) * error / 100)
            if np.random.random() < error / 100:  # Random chance based on error percentage
                # Add random error between -error and +error
                error_amount = np.random.uniform(-error, error)
                y += error_amount
        
        y_values.append(y)
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': x_values,
        'y': y_values
    })
    
    # Save to file in working directory (Union will handle persistence)
    local_path = os.path.join(current_context().working_directory, "generated_data.csv")
    df.to_csv(local_path, index=False)
    
    # Create metadata
    metadata = DataMetadata(
        x_min=x_min,
        x_max=x_max,
        error=error,
        a=a,
        b=b,
        c=c,
        num_points=len(x_values),
        formula=f"y = {a}x^2 + {b}x + {c}"
    )
    
    return FlyteFile(path=local_path), metadata


@actor.task(
    enable_deck=True
)
def visualize_data(data_file: FlyteFile, metadata: DataMetadata) -> FlyteFile:
    """
    Create visualizations of the generated dataset
    
    Args:
        data_file: Path to the CSV data file
        metadata: Dictionary containing dataset metadata
    
    Returns:
        Path to the generated visualization image
    """
    # Read data
    df = pd.read_csv(data_file)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f'Dataset Visualization\nFormula: {metadata.formula}', fontsize=16)
    
    # Scatter plot
    axes[0, 0].scatter(df['x'], df['y'], alpha=0.7, color='blue')
    axes[0, 0].set_xlabel('X Values')
    axes[0, 0].set_ylabel('Y Values')
    axes[0, 0].set_title('Data Points Scatter Plot')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Line plot
    axes[0, 1].plot(df['x'], df['y'], 'o-', color='red', alpha=0.7)
    axes[0, 1].set_xlabel('X Values')
    axes[0, 1].set_ylabel('Y Values')
    axes[0, 1].set_title('Data Points Line Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Distribution plots
    axes[1, 0].hist(df['x'], bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[1, 0].set_xlabel('X Values')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('X Values Distribution')
    
    axes[1, 1].hist(df['y'], bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 1].set_xlabel('Y Values')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Y Values Distribution')
    
    plt.tight_layout()
    
    # Save plot in working directory (Union will handle persistence)
    local_path = os.path.join(current_context().working_directory, "data_visualization.png")
    plt.savefig(local_path, dpi=300, bbox_inches='tight')
    
    # Create a deck to display the visualization in Union UI
    viz_deck = union.Deck("Data Visualization Dashboard")
    
    # Add markdown description
    description = f"""
# Dataset Visualization Report

**Formula:** {metadata.formula}

**Dataset Summary:**
- **Data Range:** x ∈ [{metadata.x_min}, {metadata.x_max}]
- **Number of Points:** {metadata.num_points}
- **Error Added:** {metadata.error}%
- **Coefficients:** a={metadata.a}, b={metadata.b}, c={metadata.c}

## Visualization Details
The plots below show:
1. **Scatter Plot:** Individual data points
2. **Line Plot:** Connected data points showing the curve
3. **X Distribution:** Histogram of input values
4. **Y Distribution:** Histogram of output values
"""
    
    viz_deck.append(MarkdownRenderer().to_html(description))
    viz_deck.append(ResponsiveImageRenderer().to_html(image_src=FlyteFile(path=local_path)))
    
    plt.close()
    
    return FlyteFile(path=local_path)


@actor.task(

    # Note: Training is NOT cached by default since hyperparameters may vary
    # Add cache=True for production models with stable hyperparameters
)
def train_mlp_model(
    data_file: FlyteFile,
    metadata: DataMetadata,
    hidden_layers: List[int] = [256, 128, 64, 32, 16],
    test_size: float = 0.2,
    random_state: int = 42,
    use_feature_engineering: bool = False,
    # Training Parameters (Real Performance Drivers)
    max_iter: int = 5000,
    learning_rate_init: float = 0.001,
    learning_rate: str = "adaptive",
    early_stopping: bool = True,
    validation_fraction: float = 0.15,
    n_iter_no_change: int = 50,
    tol: float = 1e-6,
    # Regularization Parameters
    alpha: float = 0.00001,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    # Data Processing Parameters
    use_input_scaling: bool = True,
    use_output_scaling: bool = True,
    # Local Development Flag
    save_local: bool = False
) -> Tuple[FlyteFile, FlyteFile, Dict[str, Any], Annotated[FlyteFile, ModelArtifact], Annotated[FlyteFile, DataArtifact], Annotated[FlyteFile, MetricsArtifact]]:
    """
    Train an advanced MLP model on the generated data with deep learning optimizations
    
    Args:
        data_file: Path to the CSV data file
        metadata: Dictionary containing dataset metadata including coefficients
        hidden_layers: List of hidden layer sizes (default: [256, 128, 64, 32, 16])
        test_size: Fraction of data to use for testing (default: 0.2)
        random_state: Random state for reproducibility (default: 42)
        use_feature_engineering: Whether to add polynomial features (default: False)
        
        # Training Parameters (Real Performance Drivers)
        max_iter: Maximum training iterations (default: 5000)
        learning_rate_init: Initial learning rate (default: 0.001)
        learning_rate: Learning rate schedule - 'constant'/'adaptive' (default: 'adaptive')
        early_stopping: Stop training when validation score stops improving (default: True)
        validation_fraction: Fraction of training data for validation (default: 0.15)
        n_iter_no_change: Iterations with no improvement before stopping (default: 50)
        tol: Tolerance for convergence (default: 1e-6)
        
        # Regularization Parameters
        alpha: L2 regularization strength (default: 0.00001)
        beta_1: Adam momentum parameter (default: 0.9)
        beta_2: Adam momentum parameter (default: 0.999)
        
        # Data Processing Parameters
        use_input_scaling: Whether to scale input features (default: True)
        use_output_scaling: Whether to scale output values (default: True)
    
    Returns:
        Tuple of (model file path, training metrics file path, training metrics dictionary)
    """
    # Read data
    df = pd.read_csv(data_file)
    
    # Prepare features and target
    X = df['x'].values.reshape(-1, 1)
    y = df['y'].values
    
    # Feature Engineering: Help the network by providing polynomial features
    
    if use_feature_engineering:
        # Add x^2, x^3, and other polynomial terms to help the network
        X_poly = np.column_stack([
            X.flatten(),           # x
            X.flatten() ** 2,      # x^2
            X.flatten() ** 3,      # x^3 (helps with learning curves)
            np.sin(X.flatten()),   # sin(x) (adds non-linearity)
            np.cos(X.flatten()),   # cos(x) (adds non-linearity)
        ])
        X = X_poly
        print(f"✅ Feature engineering: Extended from 1 to {X.shape[1]} features")
    
    # Conditionally normalize features for better training
    from sklearn.preprocessing import StandardScaler
    
    if use_input_scaling:
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
    else:
        scaler_X = None
        X_scaled = X
    
    if use_output_scaling:
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    else:
        scaler_y = None
        y_scaled = y
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=test_size, random_state=random_state
    )
    
    # Convert list to tuple for sklearn
    hidden_layer_sizes = tuple(hidden_layers)
    
    # Advanced MLP with configurable parameters
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',           # ReLU for deep networks
        solver='adam',               # Adam optimizer
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        learning_rate_init=learning_rate_init,
        learning_rate=learning_rate,
        alpha=alpha,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=1e-8,               # Keep epsilon fixed
        n_iter_no_change=n_iter_no_change,
        tol=tol,
        batch_size='auto',          # Let sklearn decide
        shuffle=True,               # Shuffle training data
        warm_start=False
    )
    
    print("🚀 Training advanced MLP model...")
    
    # Train the model
    model.fit(X_train, y_train)
    
    print(f"✅ Training completed in {model.n_iter_} iterations")
    
    # Make predictions (remember to inverse transform)
    y_train_pred_scaled = model.predict(X_train)
    y_test_pred_scaled = model.predict(X_test)
    
    # Inverse transform predictions back to original scale if scaling was used
    if use_output_scaling:
        y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
        y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
        # Inverse transform actual values back to original scale
        y_train_actual = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
        y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    else:
        y_train_pred = y_train_pred_scaled
        y_test_pred = y_test_pred_scaled
        y_train_actual = y_train
        y_test_actual = y_test
    
    # Calculate metrics on original scale
    train_mse = mean_squared_error(y_train_actual, y_train_pred)
    test_mse = mean_squared_error(y_test_actual, y_test_pred)
    train_r2 = r2_score(y_train_actual, y_train_pred)
    test_r2 = r2_score(y_test_actual, y_test_pred)
    
    # Additional advanced metrics
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
    train_mae = mean_absolute_error(y_train_actual, y_train_pred)
    test_mae = mean_absolute_error(y_test_actual, y_test_pred)
    
    # Save model with scalers and settings
    model_data = {
        'model': model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'use_feature_engineering': use_feature_engineering,
        'use_input_scaling': use_input_scaling,
        'use_output_scaling': use_output_scaling
    }
    
    model_path = os.path.join(current_context().working_directory, "mlp_model.joblib")
    joblib.dump(model_data, model_path)
    
    # Calculate total parameters (rough estimate for MLP)
    total_params = 0
    layer_sizes = [X.shape[1]] + list(hidden_layer_sizes) + [1]
    for i in range(len(layer_sizes) - 1):
        total_params += layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]  # weights + biases
    
    # Enhanced training metrics
    metrics = {
        'train_mse': float(train_mse),
        'test_mse': float(test_mse),
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'train_mae': float(train_mae),
        'test_mae': float(test_mae),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'hidden_layer_sizes': list(hidden_layer_sizes),
        'model_type': 'Advanced_MLPRegressor',
        'total_parameters': int(total_params),
        'n_iterations': int(model.n_iter_),
        'input_features': int(X.shape[1]),
        'feature_engineering': use_feature_engineering,
        'final_loss': float(model.loss_) if hasattr(model, 'loss_') else 'Unknown',
        'convergence_achieved': model.n_iter_ < 5000,
        # Include original data generation metadata
        'a': metadata.a,
        'b': metadata.b,
        'c': metadata.c,
        'formula': metadata.formula,
        'x_min': metadata.x_min,
        'x_max': metadata.x_max,
        'error': metadata.error
    }
    
    metrics_path = os.path.join(current_context().working_directory, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save to local outputs directory if requested (for local app development)
    if save_local:
        os.makedirs("outputs", exist_ok=True)
        
        # Copy model file
        import shutil
        local_model_path = "outputs/mlp_model.joblib"
        shutil.copy(model_path, local_model_path)
        print(f"✅ Saved model to local: {local_model_path}")
        
        # Copy metrics file
        local_metrics_path = "outputs/training_metrics.json"
        shutil.copy(metrics_path, local_metrics_path)
        print(f"✅ Saved metrics to local: {local_metrics_path}")
        
        # Copy data file for app
        local_data_path = "outputs/generated_data.csv"
        shutil.copy(data_file, local_data_path)
        print(f"✅ Saved data to local: {local_data_path}")
    
    # Create artifacts for Union AI serverless
    # Save model data as a file for artifact creation
    model_data = {
        'model': model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'use_feature_engineering': use_feature_engineering,
        'use_input_scaling': use_input_scaling,
        'use_output_scaling': use_output_scaling
    }
    
    # Save model data to file for artifact
    model_artifact_path = os.path.join(current_context().working_directory, "model_artifact.joblib")
    joblib.dump(model_data, model_artifact_path)
    
    # Create training data for artifact
    df = pd.read_csv(data_file)
    training_data = {
        'x': df['x'].tolist(),
        'y': df['y'].tolist(),
        'metadata': {
            'a': metadata.a,
            'b': metadata.b,
            'c': metadata.c,
            'formula': metadata.formula,
            'x_min': metadata.x_min,
            'x_max': metadata.x_max,
            'error': metadata.error,
            'num_points': metadata.num_points
        }
    }
    
    # Save training data to file for artifact
    training_data_artifact_path = os.path.join(current_context().working_directory, "training_data_artifact.json")
    with open(training_data_artifact_path, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    # Create metrics for artifact
    metrics_data = {
        'train_mse': float(train_mse),
        'test_mse': float(test_mse),
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'train_mae': float(train_mae),
        'test_mae': float(test_mae),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'hidden_layer_sizes': list(hidden_layer_sizes),
        'model_type': 'Advanced_MLPRegressor',
        'total_parameters': int(total_params),
        'n_iterations': int(model.n_iter_),
        'input_features': int(X.shape[1]),
        'feature_engineering': use_feature_engineering,
        'final_loss': float(model.loss_) if hasattr(model, 'loss_') else 'Unknown',
        'convergence_achieved': model.n_iter_ < 5000,
        'a': metadata.a,
        'b': metadata.b,
        'c': metadata.c,
        'formula': metadata.formula,
        'x_min': metadata.x_min,
        'x_max': metadata.x_max,
        'error': metadata.error
    }
    
    # Save metrics to file for artifact
    metrics_artifact_path = os.path.join(current_context().working_directory, "metrics_artifact.json")
    with open(metrics_artifact_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"🎯 Final Results:")
    print(f"   Train R²: {train_r2:.6f}")
    print(f"   Test R²:  {test_r2:.6f}")
    print(f"   Total Parameters: {total_params:,}")
    
    return (
        FlyteFile(path=model_path), 
        FlyteFile(path=metrics_path), 
        metrics,
        ModelArtifact.create_from(FlyteFile(path=model_artifact_path)),
        DataArtifact.create_from(FlyteFile(path=training_data_artifact_path)),
        MetricsArtifact.create_from(FlyteFile(path=metrics_artifact_path))
    )


@actor.task(
    enable_deck=True
)
def validate_model(
    model_file: FlyteFile,
    data_file: FlyteFile,
    metrics_file: FlyteFile
) -> Tuple[FlyteFile, Dict[str, Any]]:
    """
    Perform advanced validation on the trained model
    
    Args:
        model_file: Path to the trained model file
        data_file: Path to the original data file
        metrics_file: Path to the training metrics file
    
    Returns:
        Tuple of (validation report file path, validation metrics dictionary)
    """
    # Load model data (includes scalers now)
    model_data = joblib.load(model_file)
    model = model_data['model']
    scaler_X = model_data['scaler_X']
    scaler_y = model_data['scaler_y']
    use_feature_engineering = model_data['use_feature_engineering']
    
    df = pd.read_csv(data_file)
    
    with open(metrics_file, 'r') as f:
        training_metrics = json.load(f)
    
    # Prepare data with same preprocessing as training
    X = df['x'].values.reshape(-1, 1)
    y = df['y'].values
    
    # Apply same feature engineering
    if use_feature_engineering:
        X_poly = np.column_stack([
            X.flatten(),           # x
            X.flatten() ** 2,      # x^2
            X.flatten() ** 3,      # x^3
            np.sin(X.flatten()),   # sin(x)
            np.cos(X.flatten()),   # cos(x)
        ])
        X = X_poly
    
    # Apply same scaling
    X_scaled = scaler_X.transform(X)
    
    # Make predictions
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # Calculate comprehensive metrics
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
    
    full_mse = mean_squared_error(y, y_pred)
    full_r2 = r2_score(y, y_pred)
    full_mae = mean_absolute_error(y, y_pred)
    
    # Calculate residuals
    residuals = y - y_pred
    
    # Advanced validation metrics
    validation_metrics = {
        'full_dataset_mse': float(full_mse),
        'full_dataset_r2': float(full_r2),
        'full_dataset_mae': float(full_mae),
        'residual_mean': float(np.mean(residuals)),
        'residual_std': float(np.std(residuals)),
        'residual_min': float(np.min(residuals)),
        'residual_max': float(np.max(residuals)),
        'residual_abs_mean': float(np.mean(np.abs(residuals))),
        'prediction_range': {
            'min': float(np.min(y_pred)),
            'max': float(np.max(y_pred))
        },
        'actual_range': {
            'min': float(np.min(y)),
            'max': float(np.max(y))
        },
        'max_absolute_error': float(np.max(np.abs(residuals))),
        'r2_percentage': float(full_r2 * 100)
    }
    
    # Create validation visualizations
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('🚀 Model Validation Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Predictions vs Actual scatter plot
    axes[0, 0].scatter(y, y_pred, alpha=0.7, color='blue', s=50)
    axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title(f'Predictions vs Actual (R² = {full_r2:.4f})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals plot
    axes[0, 1].scatter(y_pred, residuals, alpha=0.7, color='green', s=50)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Analysis')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Residuals histogram
    axes[1, 0].hist(residuals, bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Residuals Distribution')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
    
    # 4. Model performance metrics visualization
    metrics_names = ['Train R²', 'Test R²', 'Full R²']
    metrics_values = [training_metrics['train_r2'], training_metrics['test_r2'], full_r2]
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    bars = axes[1, 1].bar(metrics_names, metrics_values, color=colors, alpha=0.8, edgecolor='black')
    axes[1, 1].set_ylabel('R² Score')
    axes[1, 1].set_title('Model Performance Comparison')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save validation visualization
    validation_plot_path = os.path.join(current_context().working_directory, "validation_dashboard.png")
    plt.savefig(validation_plot_path, dpi=300, bbox_inches='tight')
    
    # Create validation deck
    validation_deck = union.Deck("Model Validation Report")
    
    # Add validation summary
    validation_summary = f"""
# 🎯 Model Validation Results

## 🏆 Performance Summary
- **Full Dataset R²:** {full_r2:.6f} ({full_r2*100:.4f}%)
- **Target Achievement:** {'✅ Excellent (>99%)' if full_r2 > 0.99 else '⚠️ Good (>95%)' if full_r2 > 0.95 else '❌ Needs Improvement'}

## 📊 Detailed Metrics
| Metric | Train | Test | Full Dataset |
|--------|-------|------|--------------|
| **R² Score** | {training_metrics['train_r2']:.6f} | {training_metrics['test_r2']:.6f} | {full_r2:.6f} |
| **MSE** | {training_metrics['train_mse']:.6f} | {training_metrics['test_mse']:.6f} | {full_mse:.6f} |
| **MAE** | {training_metrics['train_mae']:.6f} | {training_metrics['test_mae']:.6f} | {full_mae:.6f} |

## 🔍 Residual Analysis
- **Mean Residual:** {validation_metrics['residual_mean']:.6f}
- **Residual Std:** {validation_metrics['residual_std']:.6f}
- **Max Absolute Error:** {validation_metrics['max_absolute_error']:.6f}

## 🧠 Model Architecture
- **Hidden Layers:** {training_metrics['hidden_layer_sizes']}
- **Total Parameters:** {training_metrics['total_parameters']:,}
- **Training Iterations:** {training_metrics['n_iterations']}
- **Convergence:** {'✅ Yes' if training_metrics['convergence_achieved'] else '❌ No'}

## 🎯 Prediction Quality
The model shows {'excellent' if full_r2 > 0.99 else 'good' if full_r2 > 0.95 else 'moderate'} predictive performance with an R² score of {full_r2:.4f}.
"""
    
    validation_deck.append(MarkdownRenderer().to_html(validation_summary))
    validation_deck.append(ResponsiveImageRenderer().to_html(image_src=FlyteFile(path=validation_plot_path)))
    
    plt.close()
    
    # Create enhanced validation report
    report_path = os.path.join(current_context().working_directory, "validation_report.txt")
    with open(report_path, 'w') as f:
        f.write("🚀 ADVANCED MLP Model Validation Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("🏆 PERFORMANCE SUMMARY:\n")
        f.write(f"   Full Dataset R²: {full_r2:.6f} ({full_r2*100:.4f}%)\n")
        f.write(f"   Target Achievement: {'✅ >99%' if full_r2 > 0.99 else '⚠️  <99%'}\n\n")
        
        f.write("📊 Training Metrics:\n")
        f.write(f"   Train MSE: {training_metrics['train_mse']:.6f}\n")
        f.write(f"   Test MSE: {training_metrics['test_mse']:.6f}\n")
        f.write(f"   Train R²: {training_metrics['train_r2']:.6f}\n")
        f.write(f"   Test R²: {training_metrics['test_r2']:.6f}\n")
        f.write(f"   Training Iterations: {training_metrics['n_iterations']}\n")
        f.write(f"   Total Parameters: {training_metrics['total_parameters']:,}\n\n")
        
        f.write("🔍 Full Dataset Validation:\n")
        f.write(f"   Full Dataset MSE: {full_mse:.6f}\n")
        f.write(f"   Full Dataset R²: {full_r2:.6f}\n")
        f.write(f"   Full Dataset MAE: {full_mae:.6f}\n\n")
        
        f.write("📈 Residual Analysis:\n")
        f.write(f"   Residual Mean: {validation_metrics['residual_mean']:.6f}\n")
        f.write(f"   Residual Std: {validation_metrics['residual_std']:.6f}\n")
        f.write(f"   Residual Min: {validation_metrics['residual_min']:.6f}\n")
        f.write(f"   Residual Max: {validation_metrics['residual_max']:.6f}\n")
        f.write(f"   Max Absolute Error: {validation_metrics['max_absolute_error']:.6f}\n\n")
        
        f.write("🎯 Prediction Range Analysis:\n")
        f.write(f"   Predicted Min: {validation_metrics['prediction_range']['min']:.6f}\n")
        f.write(f"   Predicted Max: {validation_metrics['prediction_range']['max']:.6f}\n")
        f.write(f"   Actual Min: {validation_metrics['actual_range']['min']:.6f}\n")
        f.write(f"   Actual Max: {validation_metrics['actual_range']['max']:.6f}\n\n")
        
        f.write("🧠 Model Architecture:\n")
        f.write(f"   Hidden Layers: {training_metrics['hidden_layer_sizes']}\n")
        f.write(f"   Input Features: {training_metrics['input_features']}\n")
        f.write(f"   Feature Engineering: {training_metrics['feature_engineering']}\n")
        f.write(f"   Convergence: {training_metrics['convergence_achieved']}\n")
    
    return FlyteFile(path=report_path), validation_metrics


@workflow
def mlp_training_workflow(
    x_min: int = 1,
    x_max: int = 100,
    error: float = 0.0,
    a: float = 1.0,
    b: float = -100.0,
    c: float = 100.0,
    hidden_layers: List[int] = [256, 128, 64, 32, 16],
    test_size: float = 0.2,
    random_state: int = 42,
    use_feature_engineering: bool = False,
    # Training Parameters (Real Performance Drivers)
    max_iter: int = 5000,
    learning_rate_init: float = 0.001,
    learning_rate: str = "adaptive",
    early_stopping: bool = True,
    validation_fraction: float = 0.15,
    n_iter_no_change: int = 50,
    tol: float = 1e-6,
    # Regularization Parameters
    alpha: float = 0.00001,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    # Data Processing Parameters
    use_input_scaling: bool = True,
    use_output_scaling: bool = True,
    save_local: bool = False
) -> Tuple[FlyteFile, FlyteFile, FlyteFile, FlyteFile, FlyteFile]:
    """
    Advanced MLP training workflow with configurable performance parameters
    
    Args:
        # Data Generation
        x_min: Minimum x value for data generation
        x_max: Maximum x value for data generation
        error: Error percentage to add to y values
        a, b, c: Coefficients for the quadratic formula
        
        # Model Architecture
        hidden_layers: List of hidden layer sizes (e.g., [100, 50] for 2 layers)
        test_size: Fraction of data for testing
        random_state: Random state for reproducibility
        use_feature_engineering: Whether to add polynomial features
        
        # Training Parameters (Real Performance Drivers)
        max_iter: Maximum training iterations
        learning_rate_init: Initial learning rate
        learning_rate: Learning rate schedule ('constant' or 'adaptive')
        early_stopping: Stop training when validation score stops improving
        validation_fraction: Fraction of training data for validation
        n_iter_no_change: Iterations with no improvement before stopping
        tol: Tolerance for convergence
        
        # Regularization Parameters  
        alpha: L2 regularization strength
        beta_1: Adam momentum parameter
        beta_2: Adam momentum parameter
        
        # Data Processing Parameters
        use_input_scaling: Whether to scale input features
        use_output_scaling: Whether to scale output values
    
    Returns:
        Tuple of (data file, visualization file, model file, metrics file, validation report)
    """
    # Task 1: Generate data
    data_file, metadata = generate_data(
        x_min=x_min,
        x_max=x_max,
        error=error,
        a=a,
        b=b,
        c=c
    )
    
    # Task 2: Visualize data
    viz_file = visualize_data(data_file=data_file, metadata=metadata)
    
    # Task 3: Train advanced MLP model
    model_file, metrics_file, training_metrics, model_artifact, data_artifact, metrics_artifact = train_mlp_model(
        data_file=data_file,
        metadata=metadata,
        hidden_layers=hidden_layers,
        test_size=test_size,
        random_state=random_state,
        use_feature_engineering=use_feature_engineering,
        # Training Parameters
        max_iter=max_iter,
        learning_rate_init=learning_rate_init,
        learning_rate=learning_rate,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        n_iter_no_change=n_iter_no_change,
        tol=tol,
        # Regularization Parameters
        alpha=alpha,
        beta_1=beta_1,
        beta_2=beta_2,
        # Data Processing Parameters
        use_input_scaling=use_input_scaling,
        use_output_scaling=use_output_scaling,
        # Local Development Flag
        save_local=save_local
    )
    
    # Task 4: Advanced validation
    validation_report, validation_metrics = validate_model(
        model_file=model_file,
        data_file=data_file,
        metrics_file=metrics_file
    )
    
    return data_file, viz_file, model_file, metrics_file, validation_report

