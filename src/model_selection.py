import json
from time import perf_counter
from dataclasses import asdict

from models.linear import LogisticRegressionModel
from models.neural_networks import MLP, CNN, AttentionCNN
from models.decision_trees import XGBoostModel
from src.config import LogisticRegressionConfig, TrainingConfig, CNNConfig, XGBoostConfig, SplitConfig, results_dir
from src.data_types import TestData
from src.data_pipeline import load_experiment_data, load_eval_data
from src.metrics import calculate_metrics
from src.visualization import visualize
from src.utils import ensure_dir

MODEL_REGISTRY = {
    'logistic_regression': (LogisticRegressionModel, LogisticRegressionConfig),
    'mlp': (MLP, TrainingConfig),
    'cnn': (CNN, CNNConfig),
    'attention_cnn': (AttentionCNN, CNNConfig),
    'xgboost': (XGBoostModel, XGBoostConfig)
}

LINEAR_REGISTRY = ['logistic_regression']
NEURAL_NETWORKS_REGISTRY = ['mlp', 'cnn', 'attention_cnn']
DECISION_TREES_REGISTRY = ['xgboost']


class Model:
    def __init__(self, model_name: str, split_config: SplitConfig, config=None):
        self.model_name = model_name
        self.split_config = split_config
        self.config = config
        self.model = None
        self.metrics = None

    def create_model(self):
        try:
            model_type, model_config = MODEL_REGISTRY[self.model_name]
        except KeyError:
            raise ValueError(
                f"Model '{self.model_name}' does not exist.\n"
                f"Available models: {MODEL_REGISTRY.keys()}"
            )

        if self.config is None:
            self.config = model_config()
        elif not isinstance(self.config, model_config):
            raise TypeError(
                f"{self.model_name} requires configuration type '{model_config.__name__}'.\n"
                f"Received configuration '{type(self.config).__name__}'."
            )

        self.model = model_type(self.config)
        return self.model

    def run_model(self):
        experiment_data, _ = load_experiment_data(self.split_config)
        if self.model_name in DECISION_TREES_REGISTRY:
            development_data = experiment_data.unscaled_development
        else:
            development_data = experiment_data.scaled_development

        results_folder_path = results_dir/self.model_name
        run_id = f"{self.model_name}_{self.split_config.split_id}.json"
        if self.model_name == "cnn" or self.model_name == "attention_cnn":
            results_folder_path = results_folder_path/f"{self.model_name}_candidate_{self.config.candidate_id}"
            run_id = f"{self.model_name}_candidate_{self.config.candidate_id}_{self.split_config.split_id}.json"
        file_path = results_folder_path/run_id

        if file_path.is_file():
            raise FileExistsError(
                f"Experimental run '{run_id}' already exists and will not be overwritten."
            )

        train_start = perf_counter()
        if self.model_name in LINEAR_REGISTRY:
            self.model.train(development_data.x_train, development_data.y_train)
        else:
            self.model.train(development_data.x_train, development_data.y_train,
                             development_data.x_valid, development_data.y_valid)
        training_seconds = perf_counter() - train_start

        predict_start = perf_counter()
        predicted_classes, predicted_probabilities = self.model.predict(development_data.x_valid)
        predict_seconds = perf_counter() - predict_start

        if self.model_name in NEURAL_NETWORKS_REGISTRY:
            basic_metrics = self.model.evaluate(development_data.x_valid, development_data.y_valid)
        else:
            basic_metrics = self.model.evaluate(development_data.y_valid, predicted_classes, predicted_probabilities)

        calculated_metrics = calculate_metrics(development_data.y_valid, predicted_classes, predicted_probabilities)
        confusion_matrix = visualize(development_data.y_valid, predicted_classes)

        runtime = {'training_time': training_seconds,
                   'prediction_time': predict_seconds}

        self.metrics = {'basic_metrics': basic_metrics,
                        'calculated_metrics': calculated_metrics,
                        'confusion_matrix': confusion_matrix,
                        'runtime_seconds': runtime}
        return self.model, self.metrics

    def save_results(self) -> dict:
        results_folder_path = results_dir/self.model_name
        run_id = f"{self.model_name}_{self.split_config.split_id}.json"
        if self.model_name == "cnn" or self.model_name == "attention_cnn":
            results_folder_path = results_folder_path/f"{self.model_name}_candidate_{self.config.candidate_id}"
            run_id = f"{self.model_name}_candidate_{self.config.candidate_id}_{self.split_config.split_id}.json"
        ensure_dir(results_folder_path)
        file_path = results_folder_path/run_id

        if file_path.is_file():
            raise FileExistsError(
                f"Experimental run '{run_id}' already exists and will not be overwritten."
            )

        results_data = {'run_id': run_id,
                        'model_name': self.model_name,
                        'split_id': self.split_config.split_id,
                        'split_seed': self.split_config.split_seed,
                        'test_fraction': self.split_config.test_fraction,
                        'validation_fraction': self.split_config.validation_fraction,
                        'model_configuration': asdict(self.config),
                        'results': {
                            'loss': self.metrics['basic_metrics']['loss'],
                            'accuracy': self.metrics['basic_metrics']["accuracy"] * 100,
                            'f1': self.metrics['calculated_metrics']['f1'],
                            'recall': self.metrics['calculated_metrics']['recall'],
                            'precision': self.metrics['calculated_metrics']['precision'],
                            'avg_precision': self.metrics['calculated_metrics']['avg_precision'],
                            'roc_auc': self.metrics['calculated_metrics']['roc_auc']
                        },
                        'runtime_seconds': {
                            'training_time': self.metrics['runtime_seconds']['training_time'],
                            'prediction_time': self.metrics['runtime_seconds']['prediction_time']
                        },
                        'confusion_matrix': [self.metrics['confusion_matrix'][0].tolist(),
                                             self.metrics['confusion_matrix'][1].tolist()]}

        with file_path.open("w", encoding="utf-8") as json_file:
            json.dump(results_data, json_file, indent=2)

        return results_data

    def final_evaluation(self):
        final_eval_split = self.split_config
        final_eval_data, scaler = load_eval_data(final_eval_split)
        test_data = final_eval_data.test_data
        if self.model_name in DECISION_TREES_REGISTRY:
            train_data = final_eval_data.unscaled_train_data
        else:
            train_data = final_eval_data.scaled_train_data
            test_data = TestData(x_test=scaler.transform(test_data.x_test), y_test=test_data.y_test)

        results_folder_path = results_dir/self.model_name
        run_id = f"{self.model_name}_{self.split_config.split_id}.json"
        if self.model_name == "cnn" or self.model_name == "attention_cnn":
            results_folder_path = results_folder_path/f"{self.model_name}_candidate_{self.config.candidate_id}"
            run_id = f"{self.model_name}_candidate_{self.config.candidate_id}_{self.split_config.split_id}.json"
        file_path = results_folder_path/run_id

        if file_path.is_file():
            raise FileExistsError(
                f"Experimental run '{run_id}' already exists and will not be overwritten."
            )

        train_start = perf_counter()
        if self.model_name in LINEAR_REGISTRY:
            self.model.train(train_data.x_test, train_data.y_test)
        else:
            self.model.train(train_data.x_test, train_data.y_test)
        training_seconds = perf_counter() - train_start

        predict_start = perf_counter()
        predicted_classes, predicted_probabilities = self.model.predict(test_data.x_test)
        predict_seconds = perf_counter() - predict_start

        if self.model_name in NEURAL_NETWORKS_REGISTRY:
            basic_metrics = self.model.evaluate(test_data.x_test, test_data.y_test)
        else:
            basic_metrics = self.model.evaluate(test_data.y_test, predicted_classes, predicted_probabilities)

        calculated_metrics = calculate_metrics(test_data.y_test, predicted_classes, predicted_probabilities)
        confusion_matrix = visualize(test_data.y_test, predicted_classes)

        runtime = {'training_time': training_seconds,
                   'prediction_time': predict_seconds}

        self.metrics = {'basic_metrics': basic_metrics,
                        'calculated_metrics': calculated_metrics,
                        'confusion_matrix': confusion_matrix,
                        'runtime_seconds': runtime}
        return self.model, self.metrics


def final_evaluation(model_name: str, final_eval_split: SplitConfig, config=None):
    model = Model(model_name, split_config=final_eval_split, config=config)
    model.create_model()
    model.final_evaluation()
    final_results = model.save_results()
    return model, final_results
