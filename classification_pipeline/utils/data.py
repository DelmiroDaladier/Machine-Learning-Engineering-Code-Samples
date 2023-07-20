from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

def create_data():
    data = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], random_state=0)
    return X_train, X_test, y_train, y_test