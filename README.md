# MLP Training Workflow for Union AI

This project implements a complete machine learning workflow using Union AI's serverless platform to train a Multi-Layer Perceptron (MLP) model on synthetic quadratic data.

## Overview

The project includes multiple workflows:

### **Basic MLP Training Workflow** (`mlp_training_workflow`)
1. **Data Generation**: Creates synthetic data using the formula `y = ax² + bx + c`
2. **Data Visualization**: Generates comprehensive visualizations of the dataset
3. **Model Training**: Trains an MLP model with configurable architecture and parameters
4. **Model Validation**: Performs comprehensive validation and analysis

### **Hyperparameter Optimization Workflow** (`hyperparameter_optimization_workflow`) 
1. **Data Generation**: Same as basic workflow
2. **Data Visualization**: Same as basic workflow  
3. **Grid Search**: Automatically tries different hyperparameter combinations to find the best

### **Complete Auto-ML Workflow** (`train_with_best_params_workflow`)
1. **Find Best Parameters**: Runs hyperparameter optimization
2. **Train Final Model**: Uses best parameters to train the final model
3. **Validation**: Comprehensive validation of the optimized model

Plus a separate Gradio app (`app.py`) for model inference.

## Project Structure

```
flyte-exponential-mlp-model/
├── workflow.py              # Main Union AI workflow definition (with artifacts)
├── gridsearchoptimize.py    # Hyperparameter optimization workflows
├── app.py                   # Gradio web application for inference
├── requirements.txt         # Python dependencies
└── README.md              # This file
```

## Features

- **Configurable Data Generation**: Customize x range, error levels, and quadratic coefficients
- **Rich Visualizations**: Multiple plots including scatter plots, line plots, and distributions
- **MLP Training**: Neural network with configurable hidden layers
- **Comprehensive Validation**: MSE, R² scores, residual analysis
- **Interactive Web Interface**: Gradio app for real-time predictions
- **Union AI Integration**: Fully compatible with Union AI serverless platform
- **Dual Deployment Support**: Works with both Union AI artifacts (serverless) and local files (development)
- **Hyperparameter Optimization**: Grid search workflows for automated model tuning

## Quick Start with Union AI CLI

### 1. Install Union AI CLI

```bash
uv tool install union
```

### 2. Login to Union AI

```bash
union create login --serverless
```

### 3. Create uv virtual environment

```bash
uv venv
```

### 4. Install Dependencies

```bash
uv pip install -r requirements.txt
```

### 5. Run the Workflow

```bash
# For Union AI serverless (creates artifacts)
union run workflow.py --save_local=False

# For local development (saves files to outputs/)
union run workflow.py --save_local=True
```

### 6. Deploy the Gradio App

```bash
# Deploy to Union AI serverless (consumes artifacts)
union deploy apps union_app.py mlp-predictor

# Or run locally (requires save_local=True)
python app.py
```

### Union.ai App Features

The Union.ai app automatically:
- Downloads model artifacts from your workflow runs
- Loads the trained MLP model and associated data
- Provides the same Gradio interface as local development
- Scales automatically based on demand
- Uses environment variables to locate model files

### Artifact Integration

The app is configured to use Union.ai artifacts:
- **Model Artifact**: Contains the trained MLP model and scalers
- **Data Artifact**: Contains the training dataset
- **Metrics Artifact**: Contains training metrics and metadata

These artifacts are automatically downloaded when the app starts and made available via environment variables.

## Workflow Parameters

### Data Generation Parameters

- `x_min` (int, default: -10): Minimum x value
- `x_max` (int, default: 10): Maximum x value
- `error` (float, default: 0.0): Error percentage to add to y values
- `a` (float, default: 1.0): Coefficient for x² term
- `b` (float, default: 2.0): Coefficient for x term
- `c` (float, default: 1.0): Constant term

### Model Training Parameters

- `hidden_layers` (list, default: [256, 128, 64, 32, 16]): Hidden layer sizes
- `test_size` (float, default: 0.2): Fraction of data for testing
- `random_state` (int, default: 42): Random state for reproducibility
- `use_feature_engineering` (bool, default: False): Whether to add polynomial features

## Workflow Tasks

### 1. Data Generation Task

Generates synthetic data using the quadratic formula `y = ax² + bx + c`:

- Creates x values in the specified range
- Calculates y values using the formula
- Adds random error if specified
- Returns CSV file and metadata

### 2. Visualization Task

Creates comprehensive visualizations:

- Scatter plot of data points
- Line plot showing trends
- Distribution histograms for x and y values
- Saves high-resolution PNG image

### 3. Model Training Task

Trains an MLP regressor:

- Splits data into train/test sets
- Configures MLP with specified hidden layers
- Trains with early stopping
- Calculates MSE and R² metrics
- Saves model and metrics

### 4. Validation Task

Performs comprehensive validation:

- Predicts on full dataset
- Calculates additional metrics
- Analyzes residuals
- Generates detailed validation report

### 5. Artifact Creation

Creates Union AI artifacts for serverless deployment:

- **ModelArtifact**: Contains trained model, scalers, and settings
- **DataArtifact**: Contains training data and metadata  
- **MetricsArtifact**: Contains training metrics and performance data

## Gradio App Features

The `app.py` file provides a web interface with:

- **Single Prediction**: Enter an X value and get the predicted Y
- **Batch Prediction**: Enter multiple X values (comma-separated) for batch predictions
- **Data Information**: View dataset statistics and model performance
- **Visualization**: Plot data points and model predictions

## Union AI CLI Commands

### Workflow Management

```bash
# List workflows
union list workflows

# Get workflow details
union get workflow mlp_training_workflow

# Delete workflow
union delete workflow mlp_training_workflow
```

### Execution Management

```bash
# List executions
union list executions

# Get execution details
union get execution <execution-id>

# Cancel execution
union cancel execution <execution-id>
```

### App Management

```bash
# List apps
union list apps

# Get app details
union get app mlp-model-predictor

# Delete app
union delete app mlp-model-predictor
```

## Example Usage

### Basic Workflow Run

```bash
# Manual hyperparameter tuning (Union AI serverless)
union run workflow.py mlp_training_workflow --save_local=False

# Manual hyperparameter tuning (local development)
union run workflow.py mlp_training_workflow --save_local=True

# Automatic hyperparameter grid search (50 trials)
union run gridsearchoptimize.py hyperparameter_optimization_workflow --n_trials 50

# Complete auto-ML: find best params + train final model
union run gridsearchoptimize.py train_with_best_params_workflow --n_trials 30
```

### Common ML Experimentation Patterns

#### Architecture Experiments
```bash
# Quick prototype - small network (serverless)
union run workflow.py mlp_training_workflow --hidden_layers '[50, 25]' --save_local=False

# Quick prototype - small network (local)
union run workflow.py mlp_training_workflow --hidden_layers '[50, 25]' --save_local=True

# Standard deep network (serverless)
union run workflow.py mlp_training_workflow --hidden_layers '[256, 128, 64, 32]' --save_local=False

# Very deep network (serverless)
union run workflow.py mlp_training_workflow --hidden_layers '[512, 256, 128, 64, 32, 16]' --save_local=False

# Single layer experiments (serverless)
union run workflow.py mlp_training_workflow --hidden_layers '[100]' --save_local=False
union run workflow.py mlp_training_workflow --hidden_layers '[50]' --save_local=False
```

#### Training Parameter Experiments (Real Performance Drivers)
```bash
# Fast training (fewer iterations)
union run workflow.py mlp_training_workflow --max_iter 500 --hidden_layers '[50]'

# Very patient training
union run workflow.py mlp_training_workflow --max_iter 10000 --n_iter_no_change 100

# Different learning rates
union run workflow.py mlp_training_workflow --learning_rate_init 0.01 --hidden_layers '[50]'
union run workflow.py mlp_training_workflow --learning_rate_init 0.0001 --hidden_layers '[50]'

# Constant vs adaptive learning rate
union run workflow.py mlp_training_workflow --learning_rate 'constant' --hidden_layers '[50]'
union run workflow.py mlp_training_workflow --learning_rate 'adaptive' --hidden_layers '[50]'

# Turn off early stopping (train to completion)
union run workflow.py mlp_training_workflow --early_stopping false --max_iter 1000 --hidden_layers '[50]'

# Stricter convergence
union run workflow.py mlp_training_workflow --tol 1e-8 --hidden_layers '[50]'
union run workflow.py mlp_training_workflow --tol 1e-4 --hidden_layers '[50]'
```

#### Regularization Experiments
```bash
# More regularization (prevent overfitting)
union run workflow.py mlp_training_workflow --alpha 0.001 --hidden_layers '[50]'

# Less regularization (allow perfect fitting)
union run workflow.py mlp_training_workflow --alpha 0.000001 --hidden_layers '[50]'

# No regularization
union run workflow.py mlp_training_workflow --alpha 0.0 --hidden_layers '[50]'
```

#### Data Processing Experiments
```bash
# No input scaling
union run workflow.py mlp_training_workflow --use_input_scaling false --hidden_layers '[50]'

# No output scaling  
union run workflow.py mlp_training_workflow --use_output_scaling false --hidden_layers '[50]'

# No scaling at all
union run workflow.py mlp_training_workflow --use_input_scaling false --use_output_scaling false --hidden_layers '[50]'

# Compare with/without feature engineering
union run workflow.py mlp_training_workflow --use_feature_engineering true --hidden_layers '[50]'
union run workflow.py mlp_training_workflow --use_feature_engineering false --hidden_layers '[50]'
```

#### Grid Search Hyperparameter Experiments
```bash
# Quick grid search (10 trials)
union run gridsearchoptimize.py hyperparameter_optimization_workflow --n_trials 10

# Thorough grid search (100 trials)  
union run gridsearchoptimize.py hyperparameter_optimization_workflow --n_trials 100

# Grid search with feature engineering
union run gridsearchoptimize.py hyperparameter_optimization_workflow --n_trials 50 --use_feature_engineering true

# Grid search with different data
union run gridsearchoptimize.py hyperparameter_optimization_workflow \
  --n_trials 30 \
  --a 2.0 --b -5.0 --c 10.0 \
  --x_min 1 --x_max 20

# Complete auto-ML pipeline
union run gridsearchoptimize.py train_with_best_params_workflow \
  --n_trials 20 \
  --a 1.5 --b -2.0 --c 3.0
```

### Custom Parameters

```bash
# Easy architecture changes - modify in one place! (serverless)
union run workflow.py mlp_training_workflow \
  --x_min -20 \
  --x_max 20 \
  --error 10.0 \
  --a 2.0 \
  --b -3.0 \
  --c 5.0 \
  --hidden_layers '[100, 50]' \
  --test_size 0.3 \
  --save_local=False

# Easy architecture changes - modify in one place! (local)
union run workflow.py mlp_training_workflow \
  --x_min -20 \
  --x_max 20 \
  --error 10.0 \
  --a 2.0 \
  --b -3.0 \
  --c 5.0 \
  --hidden_layers '[100, 50]' \
  --test_size 0.3 \
  --save_local=True

# Try different architectures easily:
# Small: --hidden_layers '[50, 25]'
# Medium: --hidden_layers '[100, 75, 50]' 
# Large: --hidden_layers '[512, 256, 128, 64]'
# Very deep: --hidden_layers '[256, 128, 64, 32, 16, 8]'
```

### Monitor Execution

```bash
# Get execution status
union get execution <execution-id>

# Stream logs
union logs <execution-id>
```

## How the Grid Search Works

Our custom grid search systematically tries different combinations by cycling through:

### **Architecture Patterns** (based on trial number % 4)
- **1 layer**: 50, 100, 150, ..., 500 neurons
- **2 layers**: [100,50], [200,100], [300,150], [400,200], [500,250] 
- **3 layers**: [128,96,64], [256,192,128], [384,288,192], [512,384,256]
- **4 layers**: [256,192,128,64], [512,384,256,128], [768,576,384,192]

### **Training Parameters** (cycling through options)
- **Max Iterations**: 1000, 2000, 5000, 10000 (cycles every 4 trials)
- **Learning Rate**: 0.0001, 0.001, 0.01 (cycles every 3 trials)  
- **Regularization**: 0.0, 0.00001, 0.0001, 0.001 (cycles every 4 trials)
- **Early Stopping**: Alternates True/False every trial
- **Tolerance**: 1e-8, 1e-6, 1e-4 (cycles every 3 trials)

**Example**: Trial 0 gets [50] + 1000 iters + 0.0001 LR + 0.0 alpha + True early_stopping + 1e-8 tol

**Note**: This is a deterministic grid search, not random search or Bayesian optimization like Optuna. Each trial number maps to a specific combination of hyperparameters, ensuring systematic coverage of the parameter space.

## Outputs

The workflow produces several outputs:

### Union AI Serverless (Artifacts)
1. **ModelArtifact**: Contains trained model, scalers, and settings
2. **DataArtifact**: Contains training data and metadata
3. **MetricsArtifact**: Contains training metrics and performance data

### Local Development (Files)
1. **Data File**: CSV containing generated x,y pairs
2. **Visualization**: PNG image with multiple plots
3. **Model File**: Joblib-serialized MLP model
4. **Metrics File**: JSON with training metrics
5. **Validation Report**: Text file with detailed analysis

### File Locations
- **Serverless**: Artifacts stored in Union AI cloud storage
- **Local**: Files saved to `outputs/` directory when `save_local=True`

## Error Handling

The workflow includes comprehensive error handling:

- Input validation for all parameters
- Graceful handling of data generation errors
- Model training error recovery
- Validation error reporting
- Detailed error messages for debugging

## Performance Considerations

- **Resource Allocation**: Tasks are configured with appropriate CPU/memory limits
- **Early Stopping**: MLP training uses early stopping to prevent overfitting
- **Efficient Data Processing**: Uses pandas and numpy for optimized operations
- **Scalable Architecture**: Designed for Union AI's serverless infrastructure

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Issues**: Adjust resource limits in task decorators
3. **Model Convergence**: Try different hidden layer sizes or learning rates
4. **Deployment Failures**: Check Union AI configuration and registry access

### Debug Mode

Run with verbose logging:

```bash
union run mlp_training_workflow --verbose
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
