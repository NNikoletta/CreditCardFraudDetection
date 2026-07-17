from src import load_data, preprocess
from src import visualization as vs
from models import CNNmodels as cnn


load_data.download_dataset()
x, y = load_data.load_dataset()
x_train, x_valid, x_test, train_label, valid_label, test_label = preprocess.prepare_data(x, y)

network = cnn.TestCNN()
history = network.train(x_train, train_label, x_valid, valid_label)
network.evaluate(x_test, test_label)
predicted_classes = network.predict(x_test)
vs.visualize(test_label, predicted_classes)