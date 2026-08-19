import numpy as np
import keras
from keras.activations import relu, softmax
from keras.models import Model
from keras.layers import Dense, Input, Reshape
from keras.layers import Conv1D
from keras.layers import Flatten
from keras.metrics import CategoricalAccuracy, F1Score
from keras.utils import to_categorical
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

from src.config import TrainingConfig, LogisticRegressionConfig
from models.attention import AttributeAttention


class LogisticRegressionModel:  # Baseline comparison for neural networks
    def __init__(self):
        config = LogisticRegressionConfig()
        self.class_weight = config.class_weight
        self.random_state = config.random_state
        self.solver = config.solver
        self.penalty = config.penalty
        self.max_iter = config.max_iter
        self.model = LogisticRegression()
        self.build_model()

    def build_model(self):
        self.model = LogisticRegression(class_weight=self.class_weight, random_state=self.random_state,
                                        solver=self.solver, penalty=self.penalty, max_iter=self.max_iter,
                                        verbose=1)

    def train(self, x_train, y_train):
        history = self.model.fit(x_train, y_train)
        return history

    def predict(self, x_test, y_test, valid=False):
        predicted_classes = self.model.predict(x_test)
        predicted_probabilities = self.model.predict_proba(x_test)
        if valid:
            valid_acc = accuracy_score(y_test, predicted_classes)
            valid_loss = log_loss(y_test, predicted_probabilities)
            print('Validation loss: ', round(valid_loss, ndigits=3))
            print(f'Validation accuracy: {round(valid_acc*100, ndigits=3)}%')
            test_metrics = {'loss': valid_loss,
                            'accuracy': valid_acc}
            return predicted_classes, predicted_probabilities, test_metrics
        return predicted_classes, predicted_probabilities


class Network:
    def __init__(self):
        config = TrainingConfig()
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.model = keras.Sequential()
        self.class_weight = None
        self.build_model()

    def build_model(self):
        raise NotImplementedError(
            "Subclass must implement its own build_model()."
        )

    def train(self, x_train, y_train, x_valid, y_valid):
        train_label = to_categorical(y_train)
        valid_label = to_categorical(y_valid)

        metrics = [CategoricalAccuracy(name='accuracy'), F1Score(name='f1_score')]
        self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                           metrics=metrics)
        self.model.summary()
        history = self.model.fit(x_train, train_label, class_weight=self.class_weight, batch_size=self.batch_size,
                                 epochs=self.epochs, verbose=1, validation_data=(x_valid, valid_label))
        return history

    def evaluate(self, x_test, y_test, valid=False):
        test_label = to_categorical(y_test)
        test_metrics = self.model.evaluate(x_test, test_label, verbose=1, return_dict=True)
        if valid:
            print('Validation loss: ', round(test_metrics['loss'], ndigits=3))
            print(f'Validation accuracy: {round(test_metrics["accuracy"]*100, ndigits=3)}%')
        return test_metrics

    def predict(self, x_test):
        predicted_probabilities = self.model.predict(x_test)
        predicted_classes = np.argmax(predicted_probabilities, axis=1)
        return predicted_classes, predicted_probabilities


class MLP(Network):
    def __init__(self):
        super().__init__()

    def build_model(self):
        input_main = Input(shape=(29,))

        main_branch = Dense(64, activation=relu)(input_main)
        main_branch = Dense(32, activation=relu)(main_branch)
        main_branch = Dense(16, activation=relu)(main_branch)
        softmax_out = Dense(2, activation=softmax)(main_branch)

        self.model = Model(inputs=input_main, outputs=softmax_out)
        return self.model


class CNN(Network):
    def __init__(self):
        super().__init__()

    def build_model(self):
        input_main = Input(shape=(29,))
        reshaped_main = Reshape((29, 1))(input_main)

        main_branch = Conv1D(8, kernel_size=29, strides=1, padding='same', activation=relu)(reshaped_main)
        main_branch = Conv1D(16, kernel_size=29, strides=1, padding='same', activation=relu)(main_branch)
        main_branch = Flatten()(main_branch)

        main_branch = Dense(16, activation=relu)(main_branch)
        softmax_out = Dense(2, activation=softmax)(main_branch)

        self.model = Model(inputs=input_main, outputs=softmax_out)
        return self.model


class AttentionCNN(Network):
    def __init__(self):
        super().__init__()

    def build_model(self):
        input_main = Input(shape=(29,))
        reshaped_main = Reshape((29, 1))(input_main)

        main_branch = Conv1D(8, kernel_size=29, strides=1, padding='same', activation=relu)(reshaped_main)
        main_branch = Conv1D(16, kernel_size=29, strides=1, padding='same', activation=relu)(main_branch)
        main_branch = AttributeAttention(kernel_size=2)(main_branch)
        main_branch = Flatten()(main_branch)

        main_branch = Dense(32, activation=relu)(main_branch)
        softmax_out = Dense(2, activation=softmax)(main_branch)

        self.model = Model(inputs=input_main, outputs=softmax_out)
        return self.model
