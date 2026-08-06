import xgboost as xgb
from sklearn.metrics import accuracy_score


def xg(x_train, y_train, x_test, y_test, x_val, y_val):
    model = xgb.XGBClassifier(objective='binary:logistic', learning_rate=0.1)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)])
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    # sn.metrics.model.score(x_test, y_test)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    return y_pred