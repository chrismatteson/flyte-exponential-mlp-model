from flytekit import workflow, task, Resources, ImageSpec, map_task
from flytekit.types.file import FlyteFile
from flytekit.types.schema import FlyteSchema
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from typing import Tuple, List, Dict, Any
import json
import os

# Grid search for hyperparameter optimization
# Note: This is a custom grid search implementation, not using Optuna

image_spec = ImageSpec(
    name="mlp-training-workflow",
    builder="union",
    registry=os.environ.get("IMAGE_SPEC_REGISTRY"),
    requirements="requirements.txt"
)

@task(
    container_image=image_spec,
    requests=Resources(cpu="1", mem="2Gi"),
    limits=Resources(cpu="1", mem="2Gi")
)
def generate_data(
    x_min: int = 0,
    x_max: int = 100,
    error: float = 0.0,
    a: float = 1.0,
    b: float = 2.0,
    c: float = 1.0
) -> Tuple[FlyteFile, Dict[str, Any]]:
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
    
    # Save to file
    output_path = "./tmp/generated_data.csv"
    df.to_csv(output_path, index=False)
    
    # Create metadata
    metadata = {
        'x_min': x_min,
        'x_max': x_max,
        'error': error,
        'a': a,
        'b': b,
        'c': c,
        'num_points': len(x_values),
        'formula': f"y = {a}x^2 + {b}x + {c}"
    }
    
    return FlyteFile(output_path), metadata


@task(
    container_image=image_spec,
    requests=Resources(cpu="1", mem="2Gi"),
    limits=Resources(cpu="1", mem="2Gi")
)
def visualize_data(data_file: FlyteFile, metadata: Dict[str, Any]) -> FlyteFile:
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
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Dataset Visualization\nFormula: {metadata["formula"]}', fontsize=16)
    
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
    
    # Save plot
    output_path = "./tmp/data_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return FlyteFile(output_path)


@task(
    requests=Resources(cpu="2", mem="4Gi"),
    limits=Resources(cpu="2", mem="4Gi")
)
def train_mlp_model(
    data_file: FlyteFile,
    metadata: Dict[str, Any],
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
    use_output_scaling: bool = True
) -> Tuple[FlyteFile, FlyteFile, Dict[str, Any]]:
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
    
    model_path = "./tmp/mlp_model.joblib"
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
        'a': metadata.get('a', 1.0),
        'b': metadata.get('b', 2.0),
        'c': metadata.get('c', 1.0),
        'formula': metadata.get('formula', 'y = x² + 2x + 1'),
        'x_min': metadata.get('x_min', 0),
        'x_max': metadata.get('x_max', 100),
        'error': metadata.get('error', 0.0)
    }
    
    metrics_path = "./tmp/training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"🎯 Final Results:")
    print(f"   Train R²: {train_r2:.6f}")
    print(f"   Test R²:  {test_r2:.6f}")
    print(f"   Total Parameters: {total_params:,}")
    
    return FlyteFile(model_path), FlyteFile(metrics_path), metrics


@task(
    requests=Resources(cpu="1", mem="2Gi"),
    limits=Resources(cpu="1", mem="2Gi")
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
    
    # Create enhanced validation report
    report_path = "./tmp/validation_report.txt"
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
    
    return FlyteFile(report_path), validation_metrics


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
    use_output_scaling: bool = True
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
    model_file, metrics_file, training_metrics = train_mlp_model(
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
        use_output_scaling=use_output_scaling
    )
    
    # Task 4: Advanced validation
    validation_report, validation_metrics = validate_model(
        model_file=model_file,
        data_file=data_file,
        metrics_file=metrics_file
    )
    
    return data_file, viz_file, model_file, metrics_file, validation_report


# Optuna Hyperparameter Optimization Tasks

@task(
    container_image=image_spec,
    requests=Resources(cpu="2", mem="4Gi"),
    limits=Resources(cpu="2", mem="4Gi")
)
def grid_search_objective(
    trial_number: int,
    data_file: FlyteFile,
    metadata: Dict[str, Any],
    test_size: float = 0.2,
    random_state: int = 42,
    use_feature_engineering: bool = False,
    # Fixed data processing (what we know works)
    #use_input_scaling: bool = True,
    #use_output_scaling: bool = True
) -> float:
    """
    Grid search objective function - trains a model with hyperparameters based on trial number
    
    Args:
        trial_number: Trial number for grid search
        data_file: Path to the CSV data file
        metadata: Dictionary containing dataset metadata
        test_size: Fraction of data for testing
        random_state: Random state for reproducibility
        use_feature_engineering: Whether to add polynomial features
        use_input_scaling: Whether to scale input features
        use_output_scaling: Whether to scale output values
    
    Returns:
        R² score to maximize
    """

    use_input_scaling = True
    use_output_scaling = True
    
    # Generate hyperparameters based on trial number (grid search approach)
    
    # Suggest hyperparameters
    # Architecture parameters
    n_layers = trial_number % 4 + 1  # 1-4 layers
    hidden_layers = []
    
    if n_layers == 1:
        hidden_layers = [int(50 * (1 + trial_number % 10))]  # 50-500
    elif n_layers == 2:
        layer1 = int(100 * (1 + trial_number % 5))  # 100-500
        layer2 = int(layer1 * 0.5)  # Half of first layer
        hidden_layers = [layer1, layer2]
    elif n_layers == 3:
        layer1 = int(128 * (1 + trial_number % 4))  # 128-512
        layer2 = int(layer1 * 0.75)
        layer3 = int(layer1 * 0.5)
        hidden_layers = [layer1, layer2, layer3]
    else:  # 4 layers
        layer1 = int(256 * (1 + trial_number % 3))  # 256-768
        layer2 = int(layer1 * 0.75)
        layer3 = int(layer1 * 0.5)
        layer4 = int(layer1 * 0.25)
        hidden_layers = [layer1, layer2, layer3, layer4]
    
    # Training parameters
    max_iter_options = [1000, 2000, 5000, 10000]
    max_iter = max_iter_options[trial_number % len(max_iter_options)]
    
    lr_options = [0.0001, 0.001, 0.01]
    learning_rate_init = lr_options[trial_number % len(lr_options)]
    
    alpha_options = [0.0, 0.00001, 0.0001, 0.001]
    alpha = alpha_options[trial_number % len(alpha_options)]
    
    early_stopping = (trial_number % 2) == 0  # Alternate true/false
    
    tol_options = [1e-8, 1e-6, 1e-4]
    tol = tol_options[trial_number % len(tol_options)]
    
    # Train model with these parameters
    try:
        # Read data
        df = pd.read_csv(data_file)
        
        # Prepare features and target
        X = df['x'].values.reshape(-1, 1)
        y = df['y'].values
        
        # Feature Engineering
        if use_feature_engineering:
            X_poly = np.column_stack([
                X.flatten(),           # x
                X.flatten() ** 2,      # x^2
                X.flatten() ** 3,      # x^3
                np.sin(X.flatten()),   # sin(x)
                np.cos(X.flatten()),   # cos(x)
            ])
            X = X_poly
        
        # Scaling
        from sklearn.preprocessing import StandardScaler
        
        if use_input_scaling:
            scaler_X = StandardScaler()
            X_scaled = scaler_X.fit_transform(X)
        else:
            X_scaled = X
        
        if use_output_scaling:
            scaler_y = StandardScaler()
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        else:
            y_scaled = y
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=test_size, random_state=random_state
        )
        
        # Train model
        model = MLPRegressor(
            hidden_layer_sizes=tuple(hidden_layers),
            activation='relu',
            solver='adam',
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=early_stopping,
            validation_fraction=0.15,
            learning_rate_init=learning_rate_init,
            learning_rate='adaptive',
            alpha=alpha,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            n_iter_no_change=50,
            tol=tol,
            batch_size='auto',
            shuffle=True,
            warm_start=False
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions on test set
        y_test_pred_scaled = model.predict(X_test)
        
        # Inverse transform if scaling was used
        if use_output_scaling:
            y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
            y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        else:
            y_test_pred = y_test_pred_scaled
            y_test_actual = y_test
        
        # Calculate R² score
        r2 = r2_score(y_test_actual, y_test_pred)
        
        print(f"Trial {trial_number}: Hidden={hidden_layers}, MaxIter={max_iter}, LR={learning_rate_init}, Alpha={alpha}, R²={r2:.6f}")
        
        return float(r2)
        
    except Exception as e:
        print(f"Trial {trial_number} failed: {str(e)}")
        return 0.0  # Return poor score for failed trials


@task(
    container_image=image_spec,
    requests=Resources(cpu="1", mem="2Gi"),
    limits=Resources(cpu="1", mem="2Gi")
)
def analyze_grid_search_results(
    trial_scores: List[float],
    n_trials: int = 50
) -> Tuple[Dict[str, Any], FlyteFile]:
    """
    Analyze grid search results and find best parameters
    
    Args:
        trial_scores: List of R² scores from all trials
        n_trials: Number of trials run
    
    Returns:
        Tuple of (best parameters dictionary, results file)
    """
    
    results = []
    best_score = -np.inf
    best_params = {}
    best_trial = 0
    
    # Analyze all trial results
    for trial_num in range(n_trials):
        score = trial_scores[trial_num] if trial_num < len(trial_scores) else 0.0
        
        # Reconstruct the parameters based on trial number (grid search pattern)
        n_layers = trial_num % 4 + 1
        hidden_layers = []
        
        if n_layers == 1:
            hidden_layers = [int(50 * (1 + trial_num % 10))]
        elif n_layers == 2:
            layer1 = int(100 * (1 + trial_num % 5))
            layer2 = int(layer1 * 0.5)
            hidden_layers = [layer1, layer2]
        elif n_layers == 3:
            layer1 = int(128 * (1 + trial_num % 4))
            layer2 = int(layer1 * 0.75)
            layer3 = int(layer1 * 0.5)
            hidden_layers = [layer1, layer2, layer3]
        else:
            layer1 = int(256 * (1 + trial_num % 3))
            layer2 = int(layer1 * 0.75)
            layer3 = int(layer1 * 0.5)
            layer4 = int(layer1 * 0.25)
            hidden_layers = [layer1, layer2, layer3, layer4]
        
        max_iter_options = [1000, 2000, 5000, 10000]
        max_iter = max_iter_options[trial_num % len(max_iter_options)]
        
        lr_options = [0.0001, 0.001, 0.01]
        learning_rate_init = lr_options[trial_num % len(lr_options)]
        
        alpha_options = [0.0, 0.00001, 0.0001, 0.001]
        alpha = alpha_options[trial_num % len(alpha_options)]
        
        early_stopping = (trial_num % 2) == 0
        
        tol_options = [1e-8, 1e-6, 1e-4]
        tol = tol_options[trial_num % len(tol_options)]
        
        trial_result = {
            'trial_number': trial_num,
            'r2_score': score,
            'hidden_layers': hidden_layers,
            'max_iter': max_iter,
            'learning_rate_init': learning_rate_init,
            'alpha': alpha,
            'early_stopping': early_stopping,
            'tol': tol,
            'use_feature_engineering': False  # Fixed for this analysis
        }
        
        results.append(trial_result)
        
        if float(score) > float(best_score):
            best_score = float(score)
            best_params = trial_result.copy()
            best_trial = trial_num
            print(f"🎯 New best score: {best_score:.6f} with {hidden_layers}")
    
    # Save results
    results_path = "./tmp/hyperparameter_search_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'best_params': best_params,
            'all_results': results,
            'n_trials': n_trials,
            'best_score': float(best_score),
            'best_trial': best_trial
        }, f, indent=2)
    
    print(f"🏆 Best parameters found:")
    print(f"   Trial: {best_trial}")
    print(f"   Hidden Layers: {best_params['hidden_layers']}")
    print(f"   Max Iterations: {best_params['max_iter']}")
    print(f"   Learning Rate: {best_params['learning_rate_init']}")
    print(f"   Alpha: {best_params['alpha']}")
    print(f"   Best R² Score: {best_score:.6f}")
    
    return best_params, FlyteFile(results_path)


@workflow
def hyperparameter_optimization_workflow(
    x_min: int = 1,
    x_max: int = 100,
    error: float = 0.0,
    a: float = 1.0,
    b: float = -100.0,
    c: float = 100.0,
    n_trials: int = 50,
    test_size: float = 0.2,
    random_state: int = 42,
    use_feature_engineering: bool = False
) -> Tuple[FlyteFile, FlyteFile, Dict[str, Any], FlyteFile]:
    """
    Hyperparameter optimization workflow using grid search
    
    Args:
        x_min: Minimum x value for data generation
        x_max: Maximum x value for data generation
        error: Error percentage to add to y values
        a, b, c: Coefficients for the quadratic formula
        n_trials: Number of hyperparameter optimization trials
        test_size: Fraction of data for testing
        random_state: Random state for reproducibility
        use_feature_engineering: Whether to add polynomial features
    
    Returns:
        Tuple of (data file, visualization file, best parameters, search results file)
    """
    #temp
    n_trials = 50
    # Task 1: Generate data (same as before)
    data_file, metadata = generate_data(
        x_min=x_min,
        x_max=x_max,
        error=error,
        a=a,
        b=b,
        c=c
    )
    
    # Task 2: Visualize data (same as before)
    viz_file = visualize_data(data_file=data_file, metadata=metadata)
    
    # Task 3: Run grid search trials in parallel
    trial_numbers = list(range(n_trials))
    
    # Create lists for broadcast parameters
    data_files = [data_file] * n_trials
    metadatas = [metadata] * n_trials
    test_sizes = [test_size] * n_trials
    random_states = [random_state] * n_trials
    use_feature_engineerings = [use_feature_engineering] * n_trials
    
    # Run all trials in parallel using map_task
    trial_scores = map_task(grid_search_objective)(
        trial_number=trial_numbers,
        data_file=data_files,
        metadata=metadatas,
        test_size=test_sizes,
        random_state=random_states,
        use_feature_engineering=use_feature_engineerings
    )
    
    # Task 4: Analyze results and find best parameters
    best_params, search_results = analyze_grid_search_results(
        trial_scores=trial_scores,
        n_trials=n_trials
    )
    
    return data_file, viz_file, best_params, search_results


@workflow  
def train_with_best_params_workflow(
    x_min: int = 1,
    x_max: int = 100,
    error: float = 0.0,
    a: float = 1.0,
    b: float = -100.0,
    c: float = 100.0,
    n_trials: int = 20,
    use_feature_engineering: bool = False
) -> Tuple[FlyteFile, FlyteFile, FlyteFile, FlyteFile, FlyteFile, Dict[str, Any]]:
    """
    Full workflow: Find best hyperparameters, then train final model
    
    Args:
        x_min: Minimum x value for data generation
        x_max: Maximum x value for data generation  
        error: Error percentage to add to y values
        a, b, c: Coefficients for the quadratic formula
        n_trials: Number of hyperparameter optimization trials
        use_feature_engineering: Whether to add polynomial features
    
    Returns:
        Tuple of (data file, visualization file, model file, metrics file, validation report, best parameters)
    """
    # Step 1: Find best hyperparameters
    data_file, viz_file, best_params, search_results = hyperparameter_optimization_workflow(
        x_min=x_min,
        x_max=x_max,
        error=error,
        a=a,
        b=b,
        c=c,
        n_trials=n_trials,
        use_feature_engineering=use_feature_engineering
    )
    
    # Step 2: Train final model with best parameters
    model_file, metrics_file, training_metrics = train_mlp_model(
        data_file=data_file,
        metadata={'a': a, 'b': b, 'c': c, 'x_min': x_min, 'x_max': x_max, 'error': error},
        hidden_layers=best_params['hidden_layers'],
        test_size=0.2,
        random_state=42,
        use_feature_engineering=best_params['use_feature_engineering'],
        max_iter=best_params['max_iter'],
        learning_rate_init=best_params['learning_rate_init'],
        learning_rate='adaptive',
        early_stopping=best_params['early_stopping'],
        validation_fraction=0.15,
        n_iter_no_change=50,
        tol=best_params['tol'],
        alpha=best_params['alpha'],
        beta_1=0.9,
        beta_2=0.999,
        use_input_scaling=True,
        use_output_scaling=True
    )
    
    # Step 3: Validate final model
    validation_report, validation_metrics = validate_model(
        model_file=model_file,
        data_file=data_file,
        metrics_file=metrics_file
    )
    
    return data_file, viz_file, model_file, metrics_file, validation_report, best_params 