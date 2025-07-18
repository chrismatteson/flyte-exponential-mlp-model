"""Union.ai app declaration for the MLP Model Predictor"""

import union
import os

# Connect via Union.Remote
remote = union.UnionRemote()

# Declare the artifacts that will be used by the app
ModelArtifact = union.Artifact(name="mlp_model_artifact")
DataArtifact = union.Artifact(name="training_data_artifact")
MetricsArtifact = union.Artifact(name="training_metrics_artifact")
#ModelArtifact = os.getenv("MLP_MODEL_PATH")
#DataArtifact = os.getenv("TRAINING_DATA_PATH")
#MetricsArtifact = os.getenv("METRICS_PATH")

# The `ImageSpec` for the container that will run the `App`.
# `union-runtime` must be declared as a dependency,
# in addition to any other dependencies needed by the app code.
image = union.ImageSpec(
    name="mlp-predictor-app",
    packages=[
        "union-runtime>=0.1.11",
        "gradio>=4.0.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "joblib>=1.3.0"
    ],
    registry=os.getenv("REGISTRY"),
)

# The `App` declaration.
# Uses the `ImageSpec` declared above.
# Includes the app.py file and runs it with Gradio
Quadratic_MLP_Predictor = union.app.App(
    name="mlp-predictor",
    container_image=image,
    inputs=[
        union.app.Input(
            value=ModelArtifact.query(),
            download=True,
            env_var="MLP_MODEL_PATH",
        ),
        union.app.Input(
            value=DataArtifact.query(),
            download=True,
            env_var="TRAINING_DATA_PATH",
        ),
        union.app.Input(
            value=MetricsArtifact.query(),
            download=True,
            env_var="METRICS_PATH",
        )
    ],
    include=["app.py"],  # Include the app file
    args="python app.py",
    port=7860,
    limits=union.Resources(cpu="2", mem="2Gi"),
    requests=union.Resources(cpu="1", mem="1Gi"),
    min_replicas=1,
    max_replicas=3,
) 