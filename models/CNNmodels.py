import numpy as np
import keras
from keras.activations import relu, softmax
from keras.layers import Dense
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Flatten
from sklearn.utils.class_weight import compute_class_weight


class Network:
    def __init__(self, batch_size=16, ep=10):
        self.batch_size = batch_size
        self.ep = ep
        self.model = keras.Sequential()
        self.build_model()

    def build_model(self):
        pass

    def train(self, x_train, train_label, x_valid, valid_label):
        train_classes = np.argmax(train_label, axis=1)
        classes = np.unique(train_classes)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_classes)  # custom weights are calculated based on the distribution of the training labels
        custom_weights = {
            int(class_label): float(weight)
            for class_label, weight in zip(classes, weights)
        }
        self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                           metrics=['accuracy'])
        self.model.summary()
        history = self.model.fit(x_train, train_label, class_weight=custom_weights, batch_size=self.batch_size,
                                 epochs=self.ep, verbose=1, validation_data=(x_valid, valid_label))
        return history

    def evaluate(self, x_test, test_label):
        test_loss, test_acc = self.model.evaluate(x_test, test_label, verbose=1)
        print('Test loss: ', test_loss)
        print('Test accuracy: ', test_acc)
        return test_loss, test_acc

    def predict(self, x_test):
        predicted_classes = self.model.predict(x_test)
        predicted_classes = np.argmax(np.round(predicted_classes), axis=1)
        return predicted_classes


class PipelineTestModel(Network):  # TEMPORARY model that is used only to validate the end-to-end pipeline
    def __init__(self, batch_size=32, ep=5):  # Not intended for true fraud detection
        super().__init__(batch_size, ep)

    def build_model(self):
        self.model = keras.Sequential([
            Conv1D(8, kernel_size=2, strides=1, padding='same', activation=relu, input_shape=(29, 1)),
            Conv1D(16, kernel_size=2, strides=1, padding='valid', activation=relu),
            MaxPooling1D(pool_size=2, padding='valid'),
            Flatten(),
            Dense(40, activation=relu),
            Dense(2, activation=softmax)
        ])

