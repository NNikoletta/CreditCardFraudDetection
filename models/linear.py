import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

from src.config import LogisticRegressionConfig


class LogisticRegressionModel:  # Baseline comparison for neural networks
    def __init__(self, config: LogisticRegressionConfig):
        self.class_weight = config.class_weight
        self.random_state = config.random_state
        self.solver = config.solver
        self.penalty = config.penalty
        self.max_iter = config.max_iter
        self.model = None
        self.build_model()

    def build_model(self) -> None:
        self.model = LogisticRegression(class_weight=self.class_weight, random_state=self.random_state,
                                        solver=self.solver, penalty=self.penalty, max_iter=self.max_iter,
                                        verbose=1)

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(x_train, y_train)

    def predict(self, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        predicted_classes = self.model.predict(x_test)
        predicted_probabilities = self.model.predict_proba(x_test)[:, 1]
        return predicted_classes, predicted_probabilities

    def evaluate(self, y_test: np.ndarray, predicted_classes: np.ndarray,
                 predicted_probabilities: np.ndarray) -> dict[str, float]:
        acc = accuracy_score(y_test, predicted_classes)
        loss = log_loss(y_test, predicted_probabilities)
        print('Loss: ', round(loss, ndigits=3))
        print(f'Accuracy: {round(acc * 100, ndigits=3)}%')
        test_metrics = {'loss': loss,
                        'accuracy': acc}
        return test_metrics
