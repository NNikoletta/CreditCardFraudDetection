import numpy as np
import keras
from keras.activations import relu, softmax
from keras.models import Model
from keras.layers import Dense, Input, Reshape
from keras.layers import Conv1D, AvgPool1D, Dropout, Concatenate
from keras.layers import Flatten
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight

from src.config import TrainingConfig


class Network:
    def __init__(self):
        config = TrainingConfig()
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.model = keras.Sequential()
        self.build_model()

    def build_model(self):
        pass

    def train(self, x_train, y_train, x_valid, y_valid):
        train_label = to_categorical(y_train)
        valid_label = to_categorical(y_valid)
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)  # custom weights are calculated based on the distribution of the training labels
        custom_weights = {
            int(class_label): float(weight)
            for class_label, weight in zip(classes, weights)
        }
        self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                           metrics=['accuracy', 'auc', 'f1_score'])
        self.model.summary()
        history = self.model.fit(x_train, train_label, class_weight=custom_weights, batch_size=self.batch_size,
                                 epochs=self.epochs, verbose=1, validation_data=(x_valid, valid_label))
        return history

    def evaluate(self, x_test, y_test):
        test_label = to_categorical(y_test)
        test_loss, test_acc, auc, _ = self.model.evaluate(x_test, test_label, verbose=1)
        print('Test loss: ', test_loss)
        print('Test accuracy: ', test_acc)
        print('AUC: ', auc)
        return test_loss, test_acc

    def predict(self, x_test):
        predicted_probabilities = self.model.predict(x_test)
        predicted_classes = np.argmax(predicted_probabilities, axis=1)
        return predicted_classes


class PipelineTestModel(Network):  # TEMPORARY model that is used only to validate the end-to-end pipeline
    def __init__(self):  # Not intended for true fraud detection
        super().__init__()

    def build_model(self):
        self.model = keras.Sequential([
            Input(shape=(29, 1)),
            Conv1D(32, kernel_size=2, strides=1, padding='same', activation=relu),
            AvgPool1D(pool_size=2, strides=2, padding='valid'),
            Flatten(),
            Dense(40, activation=relu),
            Dense(2, activation=softmax)
        ])


class CNN(Network):
    def __init__(self):
        super().__init__()

    def build_model(self):
        input_main = Input(shape=(29,))
        input_main = Reshape((29, 1))(input_main)

        left_branch = Conv1D(16, kernel_size=6, strides=4, padding='valid', activation=relu)(input_main)

        right_branch = Conv1D(4, kernel_size=4, strides=2, padding='valid', activation=relu)(input_main)
        right_branch = Conv1D(8, kernel_size=2, strides=2, padding='valid', activation=relu)(right_branch)

        main_branch = Concatenate()([left_branch, right_branch])
        main_branch = Flatten()(main_branch)

        main_branch = Dense(40, activation=relu)(main_branch)
        softmax_out = Dense(2, activation=softmax)(main_branch)

        self.model = Model(inputs=input_main, outputs=softmax_out)
        return self.model
