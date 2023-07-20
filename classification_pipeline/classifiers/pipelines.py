from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class ClassificationPipeline(object):
    def __init__(self, model=None):
        if model is not None:
            self.model = model

        self.pipeline = make_pipeline(StandardScaler(), self.model)

    def train_and_predict(self, X_train, X_test, y_train, y_test):
        return self.pipeline.fit(X_train, y_train).predict(X_test)