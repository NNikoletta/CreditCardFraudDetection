import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss

from src.config import XGBoostConfig


class XGBoostModel:
    def __init__(self):
        config = XGBoostConfig()
        self.learning_rate = config.learning_rate
        self.n_estimators = config.n_estimators
        self.max_depth = config.max_depth
        self.gamma = config.gamma
        self.random_state = config.random_state
        self.min_child_weight = config.min_child_weight
        self.ratio = config.ratio  # sqrt(expected_legitimate_count/expected_fraudulent_count)
        self.model = xgb.XGBClassifier()
        self.build_model()

    def build_model(self):
        self.model = xgb.XGBClassifier(objective='binary:logistic', learning_rate=self.learning_rate,
                                       n_estimators=self.n_estimators, max_depth=self.max_depth, gamma=self.gamma,
                                       random_state=self.random_state, min_child_weight=self.min_child_weight,
                                       scale_pos_weight=self.ratio,
                                       eval_metric=['logloss'])

    def train(self, x_train, x_valid, y_train, y_valid):
        history = self.model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)])
        return history

    def predict(self, x_test, y_test):
        predicted_classes = self.model.predict(x_test)
        predicted_probabilities = self.model.predict_proba(x_test)
        test_acc = accuracy_score(y_test, predicted_classes)
        test_loss = log_loss(y_test, predicted_probabilities)
        print('Test loss: ', round(test_loss, ndigits=3))
        print(f'Test accuracy: {round(test_acc*100, ndigits=3)}%')
        return predicted_classes, predicted_probabilities
