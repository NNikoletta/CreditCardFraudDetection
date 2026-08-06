from sklearn import model_selection as ms
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical


def prepare_data(x, y):  # Preparation of data for processing
    # skf = ms.StratifiedKFold(5, shuffle=True)

    x_train, x_test, train_label, test_label = ms.train_test_split(x, y, train_size=0.9, random_state=42, stratify=y)
    x_train, x_valid, train_label, valid_label = ms.train_test_split(x_train, train_label, test_size=0.11,
                                                                     random_state=42)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)
    x_test = scaler.transform(x_test)

    # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    # x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], 1)
    # x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    #
    # print(x_train.shape)
    # print(x_valid.shape)
    # print(x_test.shape)

    train_label = to_categorical(train_label)
    valid_label = to_categorical(valid_label)
    test_label = to_categorical(test_label)

    return x_train, x_valid, x_test, train_label, valid_label, test_label, scaler
