from models.linear import LogisticRegressionModel
from models.neural_networks import MLP, CNN, AttentionCNN
from models.decision_trees import XGBoostModel
from src.config import LogisticRegressionConfig, TrainingConfig, XGBoostConfig
from src.data_types import DevelopmentData, TestData
from src.metrics import calculate_metrics
from src.visualization import visualize

MODEL_REGISTRY = {
    'logistic_regression': (LogisticRegressionModel, LogisticRegressionConfig),
    'mlp': (MLP, TrainingConfig),
    'cnn': (CNN, TrainingConfig),
    'attention_cnn': (AttentionCNN, TrainingConfig),
    'xgboost': (XGBoostModel, XGBoostConfig)
}

LINEAR_REGISTRY = ['logistic_regression']
NEURAL_NETWORKS_REGISTRY = ['mlp', 'cnn', 'attention_cnn']
DECISION_TREES_REGISTRY = ['xgboost']


def create_model(model_name: str, config=None):
    try:
        model_type, model_config = MODEL_REGISTRY[model_name]
    except KeyError:
        raise ValueError(
            f"Model '{model_name}' does not exist.\n"
            f"Available models: {MODEL_REGISTRY.keys()}"
        )

    if config is None:
        config = model_config()
    elif not isinstance(config, model_config):
        raise TypeError(
            f"{model_name} requires configuration type '{model_config.__name__}'.\n"
            f"Received configuration '{type(config).__name__}'."
        )

    return model_type(config)


def select_and_run_model(model_name: str, development_data: DevelopmentData):
    model = create_model(model_name)

    if model_name in LINEAR_REGISTRY:
        model.train(development_data.x_train, development_data.y_train)
    else:
        model.train(development_data.x_train, development_data.x_valid,
                    development_data.y_train, development_data.y_valid)
    predicted_classes, predicted_probabilities = model.predict(development_data.x_valid)

    if model_name in NEURAL_NETWORKS_REGISTRY:
        model.evaluate(development_data.x_valid, development_data.y_valid)
    else:
        model.evaluate(development_data.y_valid, predicted_classes, predicted_probabilities)

    calculate_metrics(development_data.y_valid, predicted_classes, predicted_probabilities)
    visualize(development_data.y_valid, predicted_classes)
