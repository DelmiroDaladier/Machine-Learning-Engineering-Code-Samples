import classifiers.pipelines
from utils.data import create_data
from classifiers.models import ClassificationModels
from classifiers.pipelines import ClassificationPipeline
from definitions import MODEL_CONFIG_PATH

import pandas as pd

if __name__=='__main__':
    X_train, X_test, y_train, y_test = create_data()
    models = ClassificationModels(MODEL_CONFIG_PATH).get_models()
    for model in models:
        detector = classifiers.pipelines.ClassificationPipeline(model=model)
        result = detector.train_and_predict(X_train, X_test, y_train, y_test)