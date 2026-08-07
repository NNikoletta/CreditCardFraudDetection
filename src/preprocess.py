from sklearn import model_selection as ms
from sklearn.preprocessing import StandardScaler


def prepare_data(x, y):  # Preparation of data for processing

    x_train, x_test, y_train, y_test = ms.train_test_split(x, y, train_size=0.9, random_state=42, stratify=y)
    x_train, x_valid, y_train, y_valid = ms.train_test_split(x_train, y_train, test_size=0.11, random_state=42,
                                                             stratify=y_train)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_valid_scaled = scaler.transform(x_valid)
    x_test_scaled = scaler.transform(x_test)

    scaled_data = {'x_train_scaled': x_train_scaled,
                   'x_valid_scaled': x_valid_scaled,
                   'x_test_scaled': x_test_scaled}

    regular_data = {'x_train': x_train,
                   'x_valid': x_valid,
                   'x_test': x_test}

    labels = {'y_train': y_train,
              'y_valid': y_valid,
              'y_test': y_test}

    return scaled_data, regular_data, labels, scaler
