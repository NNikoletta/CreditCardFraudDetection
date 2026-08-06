import xgboost as xgb
from sklearn.metrics import accuracy_score

from src.config import XGBoostConfig


class XGBoostModel:
    def __init__(self):
        config = XGBoostConfig
        self.learning_rate = config.learning_rate
        self.model = xgb.XGBClassifier()
        self.build_model()

    def build_model(self):
        self.model = xgb.XGBClassifier(objective='binary:logistic', learning_rate=self.learning_rate)

    def train(self, x_train, x_valid, y_train, y_valid):
        history = self.model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)])
        return history

    def predict(self, x_test, y_test):
        predicted_classes = self.model.predict(x_test)
        accuracy = accuracy_score(y_test, predicted_classes)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        return predicted_classes
