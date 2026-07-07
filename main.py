from src import load_data
from src import preprocess

load_data.download_dataset()
x, y = load_data.load_dataset()
x_train, x_valid, x_test, train_label, valid_label, test_label = preprocess.prepare_data(x, y)

