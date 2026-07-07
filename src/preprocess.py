from sklearn.preprocessing import StandardScaler
from sklearn import model_selection as ms
from keras.utils import to_categorical


def normalize(x):
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    return x


def prepare_data(x, y):
    x_train, x_test, train_label, test_label = ms.train_test_split(x, y, train_size=0.67, random_state=0)
    x_train, x_valid, train_label, valid_label = ms.train_test_split(x_train, train_label, test_size=0.2, random_state=0)

    x_train = normalize(x_train)
    x_valid = normalize(x_valid)
    x_test = normalize(x_test)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    train_label = to_categorical(train_label)
    valid_label = to_categorical(valid_label)
    test_label = to_categorical(test_label)

    # print(x_train.shape)
    # print(train_label.shape)
    # print(x_valid.shape)
    # print(valid_label.shape)
    # print(x_test.shape)
    # print(test_label.shape)

    return x_train, x_valid, x_test, train_label, valid_label, test_label

