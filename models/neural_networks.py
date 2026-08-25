import numpy as np
import keras
from keras.activations import relu, sigmoid
from keras.models import Model
from keras.layers import Dense, Input, Reshape
from keras.layers import Conv1D
from keras.layers import Flatten
from keras.metrics import BinaryAccuracy

from src.config import TrainingConfig, CNNConfig
from models.attention import AttributeAttention


class Network:
    def __init__(self, config: TrainingConfig):
        keras.utils.set_random_seed(config.keras_random_seed)

        self.config = config
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.threshold = config.threshold
        self.model = keras.Sequential()
        self.class_weight = None

        self.build_model()

    def build_model(self):
        raise NotImplementedError(
            "Subclass must implement its own build_model()."
        )

    def train(self, x_train, x_valid, y_train, y_valid) -> None:
        metrics = [BinaryAccuracy(name='accuracy', threshold=self.threshold)]
        self.model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(),
                           metrics=metrics)
        self.model.summary()
        self.model.fit(x_train, y_train, class_weight=self.class_weight, batch_size=self.batch_size,
                       epochs=self.epochs, verbose=1, validation_data=(x_valid, y_valid))

    def predict(self, x_test) -> tuple[np.ndarray, np.ndarray]:
        predicted_probabilities = self.model.predict(x_test).ravel()
        predicted_classes = (predicted_probabilities > self.threshold).astype('int')
        return predicted_classes, predicted_probabilities

    def evaluate(self, x_test, y_test) -> dict[str, float]:
        test_metrics = self.model.evaluate(x_test, y_test, verbose=1, return_dict=True)
        print('Loss: ', round(test_metrics['loss'], ndigits=3))
        print(f'Accuracy: {round(test_metrics["accuracy"]*100, ndigits=3)}%')
        return test_metrics


class MLP(Network):
    def __init__(self, config: TrainingConfig):
        super().__init__(config)

    def build_model(self):
        input_main = Input(shape=(29,))

        main_branch = Dense(32, activation=relu)(input_main)
        main_branch = Dense(16, activation=relu)(main_branch)
        sigmoid_out = Dense(1, activation=sigmoid)(main_branch)

        self.model = Model(inputs=input_main, outputs=sigmoid_out)
        return self.model


class CNN(Network):
    def __init__(self, config: CNNConfig):
        super().__init__(config)
        self.config = config

    def build_model(self):
        input_main = Input(shape=(29,))
        reshaped_main = Reshape((29, 1))(input_main)

        main_branch = Conv1D(self.config.filters[0], kernel_size=self.config.kernel_size[0],
                             strides=self.config.strides[0], padding=self.config.padding[0],
                             activation=relu)(reshaped_main)
        main_branch = Conv1D(self.config.filters[1], kernel_size=self.config.kernel_size[1],
                             strides=self.config.strides[1], padding=self.config.padding[1],
                             activation=relu)(main_branch)
        main_branch = Flatten()(main_branch)

        main_branch = Dense(self.config.fc_units, activation=relu)(main_branch)
        # main_branch = Dense(8, activation=relu)(main_branch)
        sigmoid_out = Dense(1, activation=sigmoid)(main_branch)

        self.model = Model(inputs=input_main, outputs=sigmoid_out)
        return self.model


class AttentionCNN(Network):
    def __init__(self, config: CNNConfig):
        super().__init__(config)
        self.config = config

    def build_model(self):
        input_main = Input(shape=(29,))
        reshaped_main = Reshape((29, 1))(input_main)

        main_branch = Conv1D(self.config.filters[0], kernel_size=self.config.kernel_size[0],
                             strides=self.config.strides[0], padding=self.config.padding[0],
                             activation=relu)(reshaped_main)
        main_branch = Conv1D(self.config.filters[1], kernel_size=self.config.kernel_size[1],
                             strides=self.config.strides[1], padding=self.config.padding[1],
                             activation=relu)(main_branch)
        main_branch = AttributeAttention(kernel_size=2)(main_branch)
        main_branch = Flatten()(main_branch)

        main_branch = Dense(self.config.fc_units[0], activation=relu)(main_branch)
        sigmoid_out = Dense(1, activation=sigmoid)(main_branch)

        self.model = Model(inputs=input_main, outputs=sigmoid_out)
        return self.model
