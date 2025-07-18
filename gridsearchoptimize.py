
# Optuna Hyperparameter Optimization Tasks

from flytekit import workflow, task, Resources, ImageSpec, map_task, current_context
from flytekit.types.file import FlyteFile
from flytekit.types.schema import FlyteSchema
from union import ActorEnvironment
import union
from flytekitplugins.deck.renderer import ImageRenderer, MarkdownRenderer
import pandas as pd
from typing_extensions import Annotated

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
from typing import Tuple, List, Dict, Any, Union
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
        "markdown"
    ],
)

actor = ActorEnvironment(
    name="actor",
    ttl_seconds=30,
    container_image=image_spec,
    requests=Resources(cpu="1", mem="2Gi"),
    limits=Resources(cpu="1", mem="2Gi")
)

# Union AI Artifacts for serverless deployment
ModelArtifact = union.Artifact(name="mlp_model_artifact")
DataArtifact = union.Artifact(name="training_data_artifact") 
MetricsArtifact = union.Artifact(name="training_metrics_artifact")
ValidationArtifact = union.Artifact(name="validation_report_artifact")


@task(
    container_image=image_spec,
    requests=Resources(cpu="2", mem="4Gi"),
    limits=Resources(cpu="2", mem="4Gi")
)
def grid_search_objective(
    trial_number: int,
    data_file: FlyteFile,
    metadata: DataMetadata,
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
    results_path = os.path.join(current_context().working_directory, "hyperparameter_search_results.json")
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
    
    return best_params, FlyteFile(path=results_path)


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
        metadata=DataMetadata(a=a, b=b, c=c, x_min=x_min, x_max=x_max, error=error, num_points=(x_max-x_min+1), formula=f"y = {a}x^2 + {b}x + {c}"),
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