from sklearn.preprocessing import StandardScaler
from sklearn import model_selection as ms
from keras.utils import to_categorical


def prepare_data(x, y):  # Preparation of data for processing
    x_train, x_test, train_label, test_label = ms.train_test_split(x, y, train_size=0.67, random_state=0)
    x_train, x_valid, train_label, valid_label = ms.train_test_split(x_train, train_label, test_size=0.2, random_state=0)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)
    x_test = scaler.transform(x_test)

    train_label = to_categorical(train_label)
    valid_label = to_categorical(valid_label)
    test_label = to_categorical(test_label)

    print(x_train.shape)
    print(x_valid.shape)
    print(x_test.shape)

    return x_train, x_valid, x_test, train_label, valid_label, test_label
